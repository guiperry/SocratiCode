// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Giancarlo Erra - Altaire Limited
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TextEmbedderEmbeddingProvider } from "../../src/services/provider-textembedder.js";

function jsonResponse(status: number, data: unknown) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => data,
    text: async () => JSON.stringify(data),
  } as unknown as Response;
}

/** Build a /embed/batch response aligned to the request's texts. */
function batchResponse(texts: string[]) {
  const results = texts.map((t, i) => ({
    index: i,
    embedding: [Number(t) || (i + 1) * 1000],
    dimensions: 768,
    token_count: 1,
  }));
  return jsonResponse(200, {
    results,
    model: "landmark-lattice-v1",
    count: texts.length,
  });
}

/** Default mock: healthy /health, single /embed, and ordered /embed/batch. */
function defaultFetch(url: string, options: RequestInit = {}): Promise<Response> {
  const body = typeof options.body === "string" ? JSON.parse(options.body) : undefined;
  if (url.endsWith("/health")) {
    return Promise.resolve(
      jsonResponse(200, { status: "ok", model: "landmark-lattice-v1", dims: 768 }),
    );
  }
  if (url.endsWith("/embed") && options.method === "POST") {
    const val = body?.text ? Number(body.text) : 5000;
    return Promise.resolve(
      jsonResponse(200, {
        embedding: [val],
        dimensions: 768,
        model: "landmark-lattice-v1",
        token_count: 1,
      }),
    );
  }
  if (url.endsWith("/embed/batch") && options.method === "POST") {
    return Promise.resolve(batchResponse(body.texts));
  }
  return Promise.resolve(jsonResponse(404, { error: "not found" }));
}

describe("TextEmbedderEmbeddingProvider", () => {
  const originalEnv = { ...process.env };
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    process.env.TEXTEMBEDDER_URL = "http://localhost:9999";
    delete process.env.TEXTEMBEDDER_BIN_PATH;
    delete process.env.TEXTEMBEDDER_PORT;
    fetchMock = vi.fn(defaultFetch);
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    process.env = { ...originalEnv };
  });

  it("returns an empty array for empty input without calling the API", async () => {
    const provider = new TextEmbedderEmbeddingProvider();
    const result = await provider.embed([]);
    expect(result).toEqual([]);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("unscales fixed-point vectors to [0, 1] (divide by 10000)", async () => {
    const provider = new TextEmbedderEmbeddingProvider();
    const result = await provider.embed(["10000", "25000"]);
    expect(result).toEqual([[1], [2.5]]);
  });

  it("preserves input order across the FIFO queue", async () => {
    const provider = new TextEmbedderEmbeddingProvider();
    const result = await provider.embed(["3000", "7000", "1000"]);
    expect(result).toEqual([[0.3], [0.7], [0.1]]);
    // One /embed/batch call for three texts (below batch size 128)
    const batchCalls = fetchMock.mock.calls.filter(
      ([url]) => String(url).endsWith("/embed/batch"),
    );
    expect(batchCalls).toHaveLength(1);
  });

  it("rejects when the batch response count does not match the input count", async () => {
    fetchMock.mockImplementation((url: string, options: RequestInit = {}) => {
      if (url.endsWith("/embed/batch")) {
        // Return one fewer vector than requested
        const texts = JSON.parse(options.body).texts;
        const results = texts.slice(0, -1).map((t: string, i: number) => ({
          index: i,
          embedding: [Number(t)],
          dimensions: 768,
          token_count: 1,
        }));
        return Promise.resolve(
          jsonResponse(200, { results, model: "landmark-lattice-v1", count: results.length }),
        );
      }
      return defaultFetch(url, options);
    });

    const provider = new TextEmbedderEmbeddingProvider();
    await expect(provider.embed(["10000", "20000"])).rejects.toThrow(
      /returned 1 vectors for 2 texts/,
    );
  });

  it("rejects when a batch result has an out-of-range index", async () => {
    fetchMock.mockImplementation((url: string, options: RequestInit = {}) => {
      if (url.endsWith("/embed/batch")) {
        const texts = JSON.parse(options.body).texts;
        const results = texts.map((t: string, i: number) => ({
          index: i === 0 ? 99 : i, // out of range
          embedding: [Number(t)],
          dimensions: 768,
          token_count: 1,
        }));
        return Promise.resolve(
          jsonResponse(200, { results, model: "landmark-lattice-v1", count: texts.length }),
        );
      }
      return defaultFetch(url, options);
    });

    const provider = new TextEmbedderEmbeddingProvider();
    await expect(provider.embed(["10000", "20000"])).rejects.toThrow(/out-of-range index/);
  });

  it("rejects on duplicate batch indices to avoid silent misalignment", async () => {
    fetchMock.mockImplementation((url: string, options: RequestInit = {}) => {
      if (url.endsWith("/embed/batch")) {
        const results = [
          { index: 0, embedding: [1], dimensions: 768, token_count: 1 },
          { index: 0, embedding: [2], dimensions: 768, token_count: 1 },
        ];
        return Promise.resolve(
          jsonResponse(200, { results, model: "landmark-lattice-v1", count: 2 }),
        );
      }
      return defaultFetch(url, options);
    });

    const provider = new TextEmbedderEmbeddingProvider();
    await expect(provider.embed(["10000", "20000"])).rejects.toThrow(/duplicate index/);
  });

  it("aligns vectors to their server-provided index (reorder-safe)", async () => {
    fetchMock.mockImplementation((url: string, options: RequestInit = {}) => {
      if (url.endsWith("/embed/batch")) {
        const texts = JSON.parse(options.body).texts;
        // Return in reverse order to simulate a server that reorders results
        const results = texts
          .map((t: string, i: number) => ({
            index: i,
            embedding: [Number(t)],
            dimensions: 768,
            token_count: 1,
          }))
          .reverse();
        return Promise.resolve(
          jsonResponse(200, { results, model: "landmark-lattice-v1", count: texts.length }),
        );
      }
      return defaultFetch(url, options);
    });

    const provider = new TextEmbedderEmbeddingProvider();
    const result = await provider.embed(["10000", "20000", "30000"]);
    expect(result).toEqual([[1], [2], [3]]);
  });

  it("unscales embedSingle output", async () => {
    const provider = new TextEmbedderEmbeddingProvider();
    const result = await provider.embedSingle("40000");
    expect(result).toEqual([4]);
  });

  it("ensureReady throws when /health is not ok", async () => {
    fetchMock.mockImplementation((url: string, options: RequestInit = {}) => {
      if (url.endsWith("/health")) {
        return Promise.resolve(jsonResponse(503, { status: "down" }));
      }
      return defaultFetch(url, options);
    });

    const provider = new TextEmbedderEmbeddingProvider();
    await expect(provider.ensureReady()).rejects.toThrow(/not reachable/);
  });

  it("healthCheck reports available when reachable", async () => {
    const provider = new TextEmbedderEmbeddingProvider();
    const status = await provider.healthCheck();
    expect(status.available).toBe(true);
    expect(status.modelReady).toBe(true);
  });

  it("serializes concurrent embed calls without mixing results", async () => {
    const provider = new TextEmbedderEmbeddingProvider();
    const [r1, r2] = await Promise.all([
      provider.embed(["10000", "20000"]),
      provider.embed(["30000"]),
    ]);
    expect(r1).toEqual([[1], [2]]);
    expect(r2).toEqual([[3]]);
  });

  it("rejects the embed promise when a batch request times out", async () => {
    fetchMock.mockImplementation((url: string, options: RequestInit = {}) => {
      if (url.endsWith("/embed/batch")) {
        const err = new Error("The operation was aborted");
        err.name = "TimeoutError";
        return Promise.reject(err);
      }
      return defaultFetch(url, options);
    });

    const provider = new TextEmbedderEmbeddingProvider();
    await expect(provider.embed(["10000"])).rejects.toThrow(/embed failed/);
  });

  it("does not force process.exit(0) on SIGINT when a host handler is present", async () => {
    const exitSpy = vi
      .spyOn(process, "exit")
      .mockImplementation((() => undefined) as unknown as NodeJS.Process["exit"]);
    const hostHandler = () => {};
    process.on("SIGINT", hostHandler);

    const provider = new TextEmbedderEmbeddingProvider();
    await provider.ensureReady(); // registers cleanup + signal handlers

    process.emit("SIGINT");

    expect(exitSpy).not.toHaveBeenCalled();

    process.removeListener("SIGINT", hostHandler);
    exitSpy.mockRestore();
  });
});
