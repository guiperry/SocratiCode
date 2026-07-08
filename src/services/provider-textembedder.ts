// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Giancarlo Erra - Altaire Limited
/**
 * Text-Embedder embedding provider.
 *
 * Wraps the deterministic text-embedder Go binary (Landmark Lattice v1)
 * which produces bit-identical 768-dim int32 vectors scaled to [0, 10000].
 *
 * The provider converts these to float64 [0, 1] for Qdrant by dividing
 * by FixedPointScale (10000).
 *
 * Two modes:
 *   1. npm-package mode (default): the native binary is delivered by the
 *      `g-text-embedder` npm package (Publisher: guiperry). The provider
 *      downloads the platform-matched native binary via the package's
 *      `embedder-install` command and launches the server through the
 *      package's `embedder` launcher (`npx g-text-embedder --addr :<port>`),
 *      which resolves and spawns the downloaded native binary. No bundling of
 *      platform binaries in this repo is required.
 *   2. External mode: you run the binary yourself (e.g. via
 *      `npx g-text-embedder --addr :8089`) and set TEXTEMBEDDER_URL to point
 *      at the running HTTP API.
 *
 * Embed requests are serialised through a FIFO queue with health monitoring
 * and timeout recovery to prevent blockage. Before spawning the server the
 * provider probes the target port — if an instance is already running it
 * uses that one instead.
 *
 * Optional env:
 *   TEXTEMBEDDER_URL=http://localhost:8089   (external mode — skip launch)
 *   TEXTEMBEDDER_PORT=8089                   (port for the subprocess)
 */

import { type ChildProcess, spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { EmbeddingHealthStatus, EmbeddingProvider, EmbeddingReadinessResult } from "./embedding-types.js";
import { logger } from "./logger.js";

// ── Constants ───────────────────────────────────────────────────────────

const FIXED_POINT_SCALE = 10000;
const TEXTEMBEDDER_BATCH_SIZE = 128; // server parallelizes internally (default GOMAXPROCS workers)
const BINARY_START_TIMEOUT_MS = 10_000;
const HEALTH_POLL_MS = 200;

/** How often to log queue depth / health (ms). */
const QUEUE_MONITOR_INTERVAL_MS = 5_000;

/** Warn at this many queued embed requests. */
const QUEUE_WARN_DEPTH = 10;

/** Critical warning at this depth. */
const QUEUE_CRITICAL_DEPTH = 30;

/** Per-embed-request timeout (ms) — includes all internal batches. */
const EMBED_TIMEOUT_MS = 120_000;

/** Timeout for lightweight /health and /embed single-call requests (ms). */
const HEALTH_TIMEOUT_MS = 5_000;

/** After N consecutive timeouts kill the binary and let the queue re-spawn. */
const MAX_CONSECUTIVE_TIMEOUTS = 3;

/** npm package that delivers the native text-embedder binary. */
const NPM_PACKAGE = "g-text-embedder";

// ── Fetch helper ────────────────────────────────────────────────────────

/**
 * fetch() with an abort timeout. Without this, HTTP calls can hang
 * indefinitely if the binary is unresponsive, blocking the embed queue.
 */
async function fetchWithTimeout(
  url: string,
  timeoutMs: number,
  options: RequestInit = {},
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

// ── npm-package binary resolution ──────────────────────────────────────

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/** Repo root (src/services -> repo root). */
function repoRoot(): string {
  return path.resolve(__dirname, "..", "..");
}

/**
 * Resolve the `embedder` launcher shipped by the g-text-embedder npm package.
 * Prefers a locally installed dependency (node_modules/.bin/embedder); falls
 * back to `npx -y g-text-embedder` so it works without a global install (npx
 * fetches the package on first use).
 */
function resolveEmbedderBin(): { command: string; args: string[] } {
  const localBin = path.join(repoRoot(), "node_modules", ".bin", "embedder");
  if (fs.existsSync(localBin)) {
    return { command: localBin, args: [] };
  }
  const npx = process.platform === "win32" ? "npx.cmd" : "npx";
  return { command: npx, args: ["-y", NPM_PACKAGE] };
}

/**
 * Resolve the `embedder-install` command from the g-text-embedder npm package,
 * used to download the native binary matching the current OS/arch. Mirrors
 * resolveEmbedderBin(): prefer a local install, fall back to npx.
 */
function resolveInstallerBin(): { command: string; args: string[] } {
  const localBin = path.join(repoRoot(), "node_modules", ".bin", "embedder-install");
  if (fs.existsSync(localBin)) {
    return { command: localBin, args: [] };
  }
  const npx = process.platform === "win32" ? "npx.cmd" : "npx";
  return { command: npx, args: ["-y", NPM_PACKAGE, "embedder-install"] };
}

/**
 * Ensure the native text-embedder binary is downloaded by running the npm
 * package's `embedder-install` command. The installer fetches the binary that
 * matches the current platform into the package's install directory.
 */
function ensureNativeBinary(): void {
  const installer = resolveInstallerBin();
  const installArgs = installer.args.length > 0
    ? [...installer.args, "embedder-install"]
    : ["embedder-install"];
  logger.info("Ensuring G-Text Embedder native binary is installed", {
    command: [installer.command, ...installArgs].join(" "),
  });
  const result = spawnSync(installer.command, installArgs, {
    stdio: "inherit",
    env: process.env,
  });
  if (result.status !== 0) {
    throw new Error(
      "Failed to install the G-Text Embedder native binary via the " +
      `${NPM_PACKAGE} package. Try running '${installer.command} ${installArgs.join(" ")}' manually, ` +
      "or set TEXTEMBEDDER_URL to point at an already-running instance.",
    );
  }
}

// ── Subprocess management ──────────────────────────────────────────────

let subprocess: ChildProcess | null = null;
let subprocessUrl: string | null = null;
let binaryStarting = false;

/** Try to reach an already-running instance on the port before spawning. */
async function probePort(port: number): Promise<string | null> {
  const url = `http://localhost:${port}`;
  try {
    const resp = await fetch(`${url}/health`, { signal: AbortSignal.timeout(2000) });
    if (resp.ok) {
      logger.info("text-embedder instance already running on port", { port, url });
      subprocessUrl = url;
      return url;
    }
  } catch {
    // Nothing listening — will spawn
  }
  return null;
}

async function startBinary(port: number): Promise<string> {
  // Fast path: already know the URL
  if (subprocessUrl) return subprocessUrl;

  // Another caller is already starting the binary — wait for it instead of
  // spawning a second instance.
  if (binaryStarting) {
    return waitForBinaryStart();
  }

  // Claim the start lock synchronously, BEFORE any await, so that concurrent
  // callers cannot both slip past this guard and spawn duplicate subprocesses.
  binaryStarting = true;

  try {
    // Probe the port first — another process may already be running
    const existing = await probePort(port);
    if (existing) return existing;

    // Download the platform-matched native binary via the npm package.
    ensureNativeBinary();

    // Re-check port after install — an instance may have started meanwhile.
    const recheck = await probePort(port);
    if (recheck) return recheck;

    const launcher = resolveEmbedderBin();
    const url = `http://localhost:${port}`;
    const launchArgs = [...launcher.args, `--addr=:${port}`];
    logger.info("Starting G-Text Embedder via npm package", {
      command: [launcher.command, ...launchArgs].join(" "),
      port,
    });

    // Spawn detached on POSIX so we can kill the entire process group: the
    // launcher (`embedder`) spawns the native binary via spawnSync, so killing
    // only the launcher would orphan the native binary. Killing the group
    // (negative pid) terminates both.
    subprocess = spawn(launcher.command, launchArgs, {
      stdio: ["ignore", "pipe", "pipe"],
      detached: process.platform !== "win32",
    });

    subprocess.on("error", (err) => {
      logger.error("G-Text Embedder failed to start", { error: err.message });
      subprocess = null;
      binaryStarting = false;
    });

    subprocess.on("exit", (code, signal) => {
      logger.info("G-Text Embedder exited", { code, signal });
      subprocess = null;
      subprocessUrl = null;
      binaryStarting = false;
    });

    const logStream = (data: Buffer) => {
      for (const line of data.toString().split("\n").filter(Boolean)) {
        logger.debug(`[text-embedder] ${line}`);
      }
    };
    subprocess.stdout?.on("data", logStream);
    subprocess.stderr?.on("data", logStream);

    const deadline = Date.now() + BINARY_START_TIMEOUT_MS;
    while (Date.now() < deadline) {
      try {
        const resp = await fetchWithTimeout(`${url}/health`, HEALTH_TIMEOUT_MS);
        if (resp.ok) {
          logger.info("G-Text Embedder is ready", { url });
          subprocessUrl = url;
          return url;
        }
      } catch {
        // Not ready yet
      }
      await new Promise((r) => setTimeout(r, HEALTH_POLL_MS));
    }

    killBinary();
    throw new Error(
      `G-Text Embedder did not become ready within ${BINARY_START_TIMEOUT_MS / 1000}s. ` +
      "Check that the g-text-embedder npm package installed correctly for your platform.",
    );
  } finally {
    // Always release the start lock. On success subprocessUrl is set and the
    // fast path handles future calls; on failure waiters observe the cleared
    // lock and reject.
    binaryStarting = false;
  }
}

/**
 * Wait for an in-progress binary start to complete. Resolves with the URL once
 * the starting caller sets subprocessUrl; rejects if the start fails.
 */
function waitForBinaryStart(): Promise<string> {
  return new Promise((resolve, reject) => {
    const interval = setInterval(() => {
      if (subprocessUrl) {
        clearInterval(interval);
        resolve(subprocessUrl);
      } else if (!subprocess && !binaryStarting) {
        clearInterval(interval);
        reject(new Error("Binary failed to start"));
      }
    }, HEALTH_POLL_MS);
  });
}

function killBinary(): void {
  if (subprocess) {
    const pid = subprocess.pid;
    const isWindows = process.platform === "win32";
    try {
      if (!isWindows && pid) {
        // Kill the whole process group (launcher + native binary).
        process.kill(-pid, "SIGTERM");
      } else {
        subprocess.kill("SIGTERM");
      }
      setTimeout(() => {
        if (subprocess && pid) {
          try {
            if (!isWindows) process.kill(-pid, "SIGKILL");
            else subprocess.kill("SIGKILL");
          } catch {
            // Already gone
          }
        }
      }, 2000);
    } catch {
      // Already dead
    }
    subprocess = null;
    subprocessUrl = null;
  }
}

let cleanupRegistered = false;
function registerCleanup(): void {
  if (cleanupRegistered) return;
  cleanupRegistered = true;
  const cleanup = () => {
    killBinary();
  };
  process.on("exit", cleanup);
  // Cooperative shutdown: run our cleanup, but do NOT force an immediate exit
  // when the host (e.g. the MCP server) has registered its own signal handler.
  // Forcing process.exit(0) here would preempt the host's graceful shutdown.
  // We only exit ourselves when we are the sole listener for the signal.
  const onSignal = (sig: NodeJS.Signals) => {
    cleanup();
    if (process.listenerCount(sig) <= 1) {
      process.exit(0);
    }
  };
  process.on("SIGINT", () => onSignal("SIGINT"));
  process.on("SIGTERM", () => onSignal("SIGTERM"));
}

// ── Embed request queue ──────────────────────────────────────────────

interface EmbedRequest {
  texts: string[];
  resolve: (vectors: number[][]) => void;
  reject: (error: Error) => void;
  submittedAt: number;
  batchCount: number;
}

const embedQueue: EmbedRequest[] = [];
let queueWorkerRunning = false;
let queueMonitorTimer: ReturnType<typeof setInterval> | null = null;
let consecutiveTimeouts = 0;

function startQueueMonitor(): void {
  if (queueMonitorTimer) return;
  queueMonitorTimer = setInterval(() => {
    const depth = embedQueue.length;
    if (depth === 0) {
      consecutiveTimeouts = 0; // healthy — reset counter
      return;
    }

    const oldest = embedQueue[0];
    const elapsed = Date.now() - oldest.submittedAt;

    logger.debug("text-embedder embed queue status", {
      depth,
      oldestWaitingMs: elapsed,
      batchCount: oldest.batchCount,
    });

    if (depth >= QUEUE_CRITICAL_DEPTH) {
      logger.warn("text-embedder embed queue critically deep", {
        depth,
        oldestWaitingMs: elapsed,
      });
    } else if (depth >= QUEUE_WARN_DEPTH) {
      logger.warn("text-embedder embed queue growing", {
        depth,
        oldestWaitingMs: elapsed,
      });
    }
  }, QUEUE_MONITOR_INTERVAL_MS);
}

/**
 * Process the embed queue — one request at a time. Each request may span
 * multiple internal HTTP batches (of TEXTEMBEDDER_BATCH_SIZE).
 */
async function processQueue(): Promise<void> {
  if (queueWorkerRunning) return;
  queueWorkerRunning = true;

  while (embedQueue.length > 0) {
    const req = embedQueue.shift();
    if (!req) {
      break;
    }
    try {
      const startTime = Date.now();
      const results = await doEmbed(req.texts);
      const duration = Date.now() - startTime;

      if (duration > 10_000) {
        logger.debug("text-embedder slow embed request", {
          durationMs: duration,
          textCount: req.texts.length,
          batchCount: req.batchCount,
        });
      }

      consecutiveTimeouts = 0;
      req.resolve(results);
    } catch (err) {
      const isTimeout = err instanceof Error && (
        err.name === "TimeoutError" || err.message.includes("aborted")
      );

      if (isTimeout) {
        consecutiveTimeouts++;
        logger.error("text-embedder request timed out", {
          consecutiveTimeouts,
          maxConsecutive: MAX_CONSECUTIVE_TIMEOUTS,
        });

        if (consecutiveTimeouts >= MAX_CONSECUTIVE_TIMEOUTS) {
          logger.error(
            "text-embedder consecutive timeouts — killing binary for recovery",
          );
          killBinary(); // next resolveBaseUrl will re-spawn
          consecutiveTimeouts = 0;
        }
      }

      const msg = err instanceof Error ? err.message : String(err);
      req.reject(new Error(`text-embedder embed failed: ${msg}`));
    }
  }

  queueWorkerRunning = false;
}

/**
 * Internal: send a full set of texts through the binary in batches.
 * Used by the queue worker — not for direct external use.
 */
async function doEmbed(texts: string[]): Promise<number[][]> {
  const baseUrl = await resolveBaseUrl();
  const results: number[][] = [];

  for (let i = 0; i < texts.length; i += TEXTEMBEDDER_BATCH_SIZE) {
    const batch = texts.slice(i, i + TEXTEMBEDDER_BATCH_SIZE);

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), EMBED_TIMEOUT_MS);

    try {
      const response = await fetch(`${baseUrl}/embed/batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts: batch }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const body = await response.text().catch(() => "");
        throw new Error(
          `text-embedder /embed/batch failed (${response.status}): ${body}`,
        );
      }

      const data: BatchResponse = await response.json() as BatchResponse;

      if (!data || !Array.isArray(data.results)) {
        throw new Error(
          "text-embedder /embed/batch returned an invalid response (missing results array)",
        );
      }

      if (data.results.length !== batch.length) {
        throw new Error(
          `text-embedder /embed/batch returned ${data.results.length} vectors ` +
          `for ${batch.length} texts — vector/input misalignment`,
        );
      }

      // Realign by the server-provided index. This defends against a server
      // that reorders results, which would otherwise silently misalign every
      // vector with the wrong input text.
      const ordered = new Array<number[]>(batch.length);
      for (const r of data.results) {
        if (typeof r.index !== "number" || r.index < 0 || r.index >= batch.length) {
          throw new Error(
            `text-embedder /embed/batch returned out-of-range index ${r.index}`,
          );
        }
        if (ordered[r.index] !== undefined) {
          throw new Error(
            `text-embedder /embed/batch returned duplicate index ${r.index}`,
          );
        }
        ordered[r.index] = unscaleVector(r.embedding);
      }

      results.push(...ordered);
    } finally {
      clearTimeout(timer);
    }
  }

  return results;
}

// ── HTTP client ─────────────────────────────────────────────────────────

interface EmbedResponse {
  embedding: number[];
  dimensions: number;
  model: string;
  token_count: number;
}

interface BatchItem {
  index: number;
  embedding: number[];
  dimensions: number;
  token_count: number;
}

interface BatchResponse {
  results: BatchItem[];
  model: string;
  count: number;
}

interface HealthResponse {
  status: string;
  model: string;
  dims: number;
  timestamp: string;
}

async function resolveBaseUrl(): Promise<string> {
  const externalUrl = process.env.TEXTEMBEDDER_URL;
  if (externalUrl) return externalUrl.replace(/\/+$/, "");

  if (subprocessUrl) return subprocessUrl;

  const port = Number(process.env.TEXTEMBEDDER_PORT) || 8089;
  return startBinary(port);
}

function unscaleVector(intVec: number[]): number[] {
  return intVec.map((v) => v / FIXED_POINT_SCALE);
}

// ── Provider class ──────────────────────────────────────────────────────

export class TextEmbedderEmbeddingProvider implements EmbeddingProvider {
  readonly name = "textembedder";

  async ensureReady(): Promise<EmbeddingReadinessResult> {
    registerCleanup();
    const baseUrl = await resolveBaseUrl();

    try {
      const resp = await fetchWithTimeout(`${baseUrl}/health`, HEALTH_TIMEOUT_MS);
      if (!resp.ok) {
        throw new Error(`Health check returned status ${resp.status}`);
      }
      const health: HealthResponse = await resp.json() as HealthResponse;
      logger.info("text-embedder provider ready", {
        model: health.model,
        dims: health.dims,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      throw new Error(
        `text-embedder is not reachable at ${baseUrl}. ` +
        "Make sure the g-text-embedder npm package installed the binary (run " +
        "'npx g-text-embedder embedder-install'), or set TEXTEMBEDDER_URL to " +
        `point at an external instance. Underlying error: ${message}`,
      );
    }

    return { modelPulled: false, containerStarted: false, imagePulled: false };
  }

  /**
   * Embed an array of texts. Requests are serialised through a FIFO queue
   * with health monitoring and timeout recovery. The queue worker processes
   * one request at a time (each split into internal HTTP batches of
   * TEXTEMBEDDER_BATCH_SIZE).
   */
  async embed(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];

    startQueueMonitor();

    return new Promise<number[][]>((resolve, reject) => {
      embedQueue.push({
        texts,
        resolve,
        reject,
        submittedAt: Date.now(),
        batchCount: Math.ceil(texts.length / TEXTEMBEDDER_BATCH_SIZE),
      });

      // Kick off the worker (no-op if already running)
      processQueue();
    });
  }

  /**
   * Embed a single text directly (no queue). Lightweight — one /embed call.
   */
  async embedSingle(text: string): Promise<number[]> {
    const baseUrl = await resolveBaseUrl();

    const response = await fetchWithTimeout(`${baseUrl}/embed`, HEALTH_TIMEOUT_MS, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      const body = await response.text().catch(() => "");
      throw new Error(
        `text-embedder /embed failed (${response.status}): ${body}`,
      );
    }

    const data: EmbedResponse = await response.json() as EmbedResponse;
    return unscaleVector(data.embedding);
  }

  async healthCheck(): Promise<EmbeddingHealthStatus> {
    const lines: string[] = [];
    const icon = (ok: boolean) => (ok ? "[OK]" : "[MISSING]");

    const externalUrl = process.env.TEXTEMBEDDER_URL;

    if (externalUrl) {
      lines.push(`${icon(true)} text-embedder mode: external (${externalUrl})`);
    } else {
      const launcher = resolveEmbedderBin();
      lines.push(`${icon(true)} text-embedder mode: npm package (${NPM_PACKAGE})`);
      lines.push(`${icon(true)} text-embedder launcher: ${launcher.command}`);
      lines.push(`${icon(true)} text-embedder install: ${resolveInstallerBin().command}`);
    }

    try {
      const baseUrl = await resolveBaseUrl();
      const resp = await fetchWithTimeout(`${baseUrl}/health`, HEALTH_TIMEOUT_MS);
      if (!resp.ok) {
        lines.push(`${icon(false)} text-embedder: Health check failed (status ${resp.status})`);
        return { available: false, modelReady: false, statusLines: lines };
      }
      const health: HealthResponse = await resp.json() as HealthResponse;
      lines.push(`${icon(true)} text-embedder: Reachable at ${baseUrl}`);

      // Add queue info
      const queueDepth = embedQueue.length;
      if (queueDepth > 0) {
        const oldest = embedQueue[0];
        const elapsed = Date.now() - oldest.submittedAt;
        lines.push(`${icon(true)} Embed queue: ${queueDepth} pending (oldest ${elapsed}ms)`);
      } else {
        lines.push(`${icon(true)} Embed queue: idle`);
      }

      lines.push(`${icon(true)} Model: ${health.model} (${health.dims} dims)`);
      return { available: true, modelReady: true, statusLines: lines };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      lines.push(`${icon(false)} text-embedder: Not reachable (${message})`);
      return { available: false, modelReady: false, statusLines: lines };
    }
  }
}
