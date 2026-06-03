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
 *   1. Binary mode (default): the binary is distributed as text-embedder.gz.
 *      The provider decompresses it to a temp directory on first use and
 *      spawns it as a subprocess.
 *   2. External mode: user runs the binary themselves and sets TEXTEMBEDDER_URL
 *      pointing at the HTTP API (e.g. http://localhost:8089).
 *
 * Embed requests are serialised through a FIFO queue with health monitoring
 * and timeout recovery to prevent blockage. Before spawning the binary the
 * provider probes the target port — if an instance is already running it
 * uses that one instead.
 *
 * Optional env:
 *   TEXTEMBEDDER_URL=http://localhost:8089     (external mode — skip binary)
 *   TEXTEMBEDDER_BIN_PATH=<path-to-binary>     (uncompressed binary path)
 *   TEXTEMBEDDER_PORT=8089                    (port for subprocess)
 */

import { spawn, type ChildProcess } from "node:child_process";
import fsp from "node:fs/promises";
import { constants } from "node:fs";
import path from "node:path";
import os from "node:os";
import { fileURLToPath } from "node:url";
import { gunzipSync } from "node:zlib";
import { getEmbeddingConfig } from "./embedding-config.js";
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

/** After N consecutive timeouts kill the binary and let the queue re-spawn. */
const MAX_CONSECUTIVE_TIMEOUTS = 3;

/** Temp dir where the decompressed binary lives. */
const TMP_DIR = path.join(os.tmpdir(), "socraticode-textembedder");
const TMP_BIN_PATH = path.join(TMP_DIR, "text-embedder");

// ── Binary path discovery ──────────────────────────────────────────────

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PLATFORM_MAP: Record<string, string> = {
  linux: "text-embedder-linux.gz",
  darwin: "text-embedder-darwin.gz",
  win32: "text-embedder-win.gz",
};

function platformGzName(): string | null {
  return PLATFORM_MAP[process.platform] ?? null;
}

/**
 * Candidates for the gzipped binary, checked in order.
 * Platform-specific binaries take priority over the generic fallback name.
 */
function gzCandidates(): string[] {
  const candidates: string[] = [];

  const envBin = process.env.TEXTEMBEDDER_BIN_PATH;
  if (envBin) candidates.push(envBin);

  const pfx = platformGzName();
  if (pfx) {
    candidates.push(
      path.resolve(process.cwd(), pfx),
      path.resolve(__dirname, pfx),
      path.resolve(__dirname, "..", pfx),
      path.resolve(__dirname, "..", "..", pfx),
    );
  }

  candidates.push(
    path.resolve(process.cwd(), "text-embedder"),
    path.resolve(process.cwd(), "text-embedder.gz"),
    path.resolve(__dirname, "text-embedder.gz"),
    path.resolve(__dirname, "..", "text-embedder.gz"),
    path.resolve(__dirname, "..", "..", "text-embedder.gz"),
  );

  return candidates;
}

/**
 * Resolve the gzipped (or bare) binary path. Returns { gzPath, isCompressed }.
 * Returns null if nothing is found.
 */
async function resolveBinarySource(): Promise<{ sourcePath: string; isCompressed: boolean } | null> {
  for (const candidate of gzCandidates()) {
    try {
      await fsp.access(candidate, constants.R_OK);
      const isCompressed = candidate.endsWith(".gz");
      return { sourcePath: candidate, isCompressed };
    } catch {
      // not here
    }
  }
  return null;
}

// ── Decompression ──────────────────────────────────────────────────────

/** Path to an already-decompressed cached binary. */
let extractedBinPath: string | null = null;

/**
 * Ensure the decompressed binary exists in the temp dir, extracting it from
 * the gzipped package asset if needed. Returns the path to the binary.
 */
async function ensureBinaryExtracted(sourcePath: string): Promise<string> {
  // Fast path: already decompressed and file still exists (and is executable)
  if (extractedBinPath) {
    try {
      await fsp.access(extractedBinPath, constants.X_OK);
      return extractedBinPath;
    } catch {
      extractedBinPath = null;
    }
  }

  // Before overwriting, check if the temp binary already exists and is executable.
  // If it does, use it as-is to avoid ETXTBSY (text file busy) when the binary is
  // currently running as a subprocess.
  try {
    await fsp.access(TMP_BIN_PATH, constants.X_OK);
    logger.info("Using existing decompressed binary (avoiding overwrite)", { path: TMP_BIN_PATH });
    extractedBinPath = TMP_BIN_PATH;
    return extractedBinPath;
  } catch {
    // Not present — proceed with decompression
  }

  logger.info("Decompressing text-embedder binary", { source: sourcePath });

  // Read gzipped data and decompress
  const compressed = await fsp.readFile(sourcePath);
  const decompressed = gunzipSync(compressed);

  // Ensure temp dir exists
  await fsp.mkdir(TMP_DIR, { recursive: true });

  // Write with executable permissions
  await fsp.writeFile(TMP_BIN_PATH, decompressed, { mode: 0o755 });

  extractedBinPath = TMP_BIN_PATH;
  return extractedBinPath;
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

  // Probe the port first — another process may already be running
  const existing = await probePort(port);
  if (existing) return existing;

  // Another caller is already starting the binary
  if (binaryStarting) {
    return new Promise((resolve, reject) => {
      const interval = setInterval(() => {
        if (subprocessUrl) {
          clearInterval(interval);
          resolve(subprocessUrl);
        }
        if (!subprocess && !binaryStarting) {
          clearInterval(interval);
          reject(new Error("Binary failed to start"));
        }
      }, HEALTH_POLL_MS);
    });
  }

  binaryStarting = true;
  const source = await resolveBinarySource();

  if (!source) {
    binaryStarting = false;
    const pfx = platformGzName();
    throw new Error(
      `text-embedder binary not found for platform "${process.platform}". ` +
      `Run 'make deploy-all' from the text-embedder directory to generate ` +
      (pfx ? `${pfx} (expected name), ` : "") +
      "or set TEXTEMBEDDER_BIN_PATH / TEXTEMBEDDER_URL.",
    );
  }

  // Re-check port after resolving source — instance may have started while we looked
  const recheck = await probePort(port);
  if (recheck) {
    binaryStarting = false;
    return recheck;
  }

  // Decompress if necessary, or use bare binary directly
  const binPath = source.isCompressed
    ? await ensureBinaryExtracted(source.sourcePath)
    : source.sourcePath;

  const url = `http://localhost:${port}`;
  logger.info("Starting text-embedder binary", { binPath, port });

  subprocess = spawn(binPath, [`--addr=:${port}`], {
    stdio: ["ignore", "pipe", "pipe"],
    detached: false,
  });

  subprocess.on("error", (err) => {
    logger.error("text-embedder binary failed to start", { error: err.message });
    subprocess = null;
    binaryStarting = false;
  });

  subprocess.on("exit", (code, signal) => {
    logger.info("text-embedder binary exited", { code, signal });
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
      const resp = await fetch(`${url}/health`);
      if (resp.ok) {
        logger.info("text-embedder binary is ready", { url });
        subprocessUrl = url;
        binaryStarting = false;
        return url;
      }
    } catch {
      // Not ready yet
    }
    await new Promise((r) => setTimeout(r, HEALTH_POLL_MS));
  }

  killBinary();
  binaryStarting = false;
  throw new Error(
    `text-embedder binary did not become ready within ${BINARY_START_TIMEOUT_MS / 1000}s. ` +
    "Check the binary is compatible with your system.",
  );
}

function killBinary(): void {
  if (subprocess) {
    try {
      subprocess.kill("SIGTERM");
      setTimeout(() => {
        if (subprocess) {
          try { subprocess.kill("SIGKILL"); } catch { /* ignore */ }
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
    // Best-effort cleanup of the temp binary
    fsp.rm(TMP_BIN_PATH, { force: true }).catch(() => {});
    fsp.rm(TMP_DIR, { force: true, recursive: true }).catch(() => {});
  };
  process.on("exit", cleanup);
  process.on("SIGINT", () => { cleanup(); process.exit(0); });
  process.on("SIGTERM", () => { cleanup(); process.exit(0); });
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

function stopQueueMonitor(): void {
  if (queueMonitorTimer) {
    clearInterval(queueMonitorTimer);
    queueMonitorTimer = null;
  }
}

/**
 * Process the embed queue — one request at a time. Each request may span
 * multiple internal HTTP batches (of TEXTEMBEDDER_BATCH_SIZE).
 */
async function processQueue(): Promise<void> {
  if (queueWorkerRunning) return;
  queueWorkerRunning = true;

  while (embedQueue.length > 0) {
    const req = embedQueue.shift()!;
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
      results.push(...data.results.map((r) => unscaleVector(r.embedding)));
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
      const resp = await fetch(`${baseUrl}/health`);
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
        "Make sure the binary is running or set TEXTEMBEDDER_URL to point at an external instance. " +
        `Underlying error: ${message}`,
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

    const response = await fetch(`${baseUrl}/embed`, {
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
      const source = await resolveBinarySource();
      const binaryOk = !!source;
      lines.push(
        `${icon(binaryOk)} text-embedder binary: ` +
        (binaryOk
          ? `Found at ${source!.sourcePath}${source!.isCompressed ? " (gzipped)" : ""}`
          : `Not found — run 'make deploy-all' from text-embedder dir`),
      );
      if (!binaryOk) {
        return { available: false, modelReady: false, statusLines: lines };
      }
      lines.push(`${icon(true)} text-embedder mode: binary (decompresses on first use)`);
    }

    try {
      const baseUrl = await resolveBaseUrl();
      const resp = await fetch(`${baseUrl}/health`);
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
