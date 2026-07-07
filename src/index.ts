#!/usr/bin/env node
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Giancarlo Erra - Altaire Limited

// Pre-flight: refuse to start on Node versions known to break @qdrant/js-client-rest.
// The qdrant client pins undici ^6 and constructs an undici.Agent it passes to Node's
// built-in fetch() as a dispatcher. Node 26+ ships a stricter undici whose dispatcher
// hook validation rejects the v6 Agent's contract — surfaces as
// `UND_ERR_INVALID_ARG: invalid onError method` on the first qdrant request.
// (The imports below are evaluated before this check at runtime per ESM semantics,
// but qdrant-js's module-init is side-effect-light — only an actual request triggers
// the undici path — so exiting here is enough to spare users the opaque error later.)
// Tracked upstream: https://github.com/qdrant/qdrant-js/issues/134
// Upstream PRs under discussion: qdrant/qdrant-js#123 (undici major upgrade) and
// qdrant/qdrant-js#128 (inject fetch into REST transport). If either lands — or any
// other fix supersedes them — raise the upper bound in package.json's `engines.node`
// and remove this check.
const nodeMajor = Number.parseInt(process.versions.node.split(".")[0], 10);
if (Number.isFinite(nodeMajor) && nodeMajor >= 26) {
  // fs.writeSync(2, …) is the canonical Node idiom for "print fatal error then die":
  // blocking (no truncation when stderr is piped — every MCP host pipes stderr) and
  // synchronous (so process.exit(1) runs before any further top-level code).
  const msg =
    `socraticode: Node ${process.versions.node} is not supported.\n` +
    "  @qdrant/js-client-rest is incompatible with the undici bundled in Node 26+.\n" +
    "  Use Node 22.x (via nvm: `nvm install 22 && nvm use 22`, or `brew install node@22` on macOS).\n" +
    "  See https://github.com/qdrant/qdrant-js/issues/134.\n";
  writeSync(2, msg);
  process.exit(1);
}

import { randomUUID } from "node:crypto";
import { writeSync } from "node:fs";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { z } from "zod";
import { SOCRATICODE_VERSION } from "./constants.js";
import { logger, setMcpLogSender } from "./services/logger.js";
import { autoResumeIndexedProjects, gracefulShutdown } from "./services/startup.js";
import { handleContextTool } from "./tools/context-tools.js";
import { handleGraphTool } from "./tools/graph-tools.js";
import { handleIndexTool } from "./tools/index-tools.js";
import { handleManageTool } from "./tools/manage-tools.js";
import { handleQueryTool } from "./tools/query-tools.js";

// ── Transport mode ───────────────────────────────────────────────────────
// When SOCRATICODE_PORT is set, SocratiCode starts an HTTP server that
// serves multiple MCP clients via the Streamable HTTP transport (each client
// gets its own session). When unset (default), the classic stdio transport
// is used for single-client operation.
const SOCRATICODE_PORT = (() => {
  const raw = process.env.SOCRATICODE_PORT;
  if (!raw) return null;
  const port = parseInt(raw, 10);
  if (!Number.isFinite(port) || port < 1 || port > 65535) {
    throw new Error(
      `Invalid SOCRATICODE_PORT: "${raw}". Must be a number between 1 and 65535.`,
    );
  }
  return port;
})();

const server = new McpServer(
  {
    name: "socraticode",
    version: SOCRATICODE_VERSION,
  },
  {
    capabilities: {
      tools: {},
    },
  },
);

// ── Index tools ──────────────────────────────────────────────────────────

// Tool registration extracted so HTTP mode can create per-session servers.
function registerAllTools(srv: McpServer): void {
srv.tool(
  "codebase_index",
  "Start indexing a codebase in the background. Returns immediately. Call codebase_status to poll progress until 100%. Do NOT search until indexing is complete. If already indexing, returns current progress.",
  {
    projectPath: z
      .string()
      .describe("Absolute path to the project directory. If omitted, uses the current working directory.")
      .optional(),
    extraExtensions: z
      .string()
      .describe("Comma-separated list of additional file extensions to index beyond the built-in set (e.g. '.tpl,.blade,.hbs'). Useful for projects with non-standard file extensions. Can also be set globally via EXTRA_EXTENSIONS env var.")
      .optional(),
  },
  async (args) => ({
    content: [{ type: "text", text: await handleIndexTool("codebase_index", args) }],
  }),
);

srv.tool(
  "codebase_index_and_watch",
  "Start indexing a codebase and automatically start file watching once complete. Blocks until indexing finishes (may take many minutes). Calls codebase_watch after completion so future changes are tracked automatically. Prefer this over codebase_index when you want to index once and forget about it.",
  {
    projectPath: z
      .string()
      .describe("Absolute path to the project directory. If omitted, uses the current working directory.")
      .optional(),
    extraExtensions: z
      .string()
      .describe("Comma-separated list of additional file extensions to index beyond the built-in set (e.g. '.tpl,.blade,.hbs'). Useful for projects with non-standard file extensions. Can also be set globally via EXTRA_EXTENSIONS env var.")
      .optional(),
  },
  async (args) => ({
    content: [{ type: "text", text: await handleIndexTool("codebase_index_and_watch", args) }],
  }),
);

srv.tool(
  "codebase_update",
  "Incrementally update an existing codebase index. Only re-indexes changed files. Runs synchronously. Usually not needed if file watcher is active.",
  {
    projectPath: z
      .string()
      .describe("Absolute path to the project directory.")
      .optional(),
    extraExtensions: z
      .string()
      .describe("Comma-separated list of additional file extensions to index (e.g. '.tpl,.blade').")
      .optional(),
  },
  async (args) => ({
    content: [{ type: "text", text: await handleIndexTool("codebase_update", args) }],
  }),
);

srv.tool(
  "codebase_remove",
  "Remove a project's codebase index entirely from the vector database. Safely stops the file watcher, cancels any in-progress indexing/update (with drain), and waits for any in-flight graph build before deleting.",
  {
    projectPath: z.string().describe("Absolute path to the project directory."),
  },
  async (args) => ({
    content: [{ type: "text", text: await handleIndexTool("codebase_remove", args) }],
  }),
);

srv.tool(
  "codebase_stop",
  "Gracefully stop an in-progress indexing operation. The current batch will finish and checkpoint, preserving all progress. Re-run codebase_index to resume from where it left off.",
  {
    projectPath: z
      .string()
      .describe("Absolute path to the project directory. If omitted, uses the current working directory.")
      .optional(),
  },
  async (args) => ({
    content: [{ type: "text", text: await handleIndexTool("codebase_stop", args) }],
  }),
);

srv.tool(
  "codebase_watch",
  "Start/stop watching a project directory for file changes and automatically update the index. When starting, first runs an incremental update to catch any changes made since the last session, then keeps the index up to date via debounced file system watching.",
  {
    projectPath: z
      .string()
      .describe("Absolute path to the project directory.")
      .optional(),
    action: z.enum(["start", "stop", "status"]).describe("start/stop watching, or get status of watchers."),
  },
  async (args) => ({
    content: [{ type: "text", text: await handleIndexTool("codebase_watch", args) }],
  }),
);

srv.tool(
  "codebase_index_remaining",
  "Scan a directory for projects not yet indexed and optionally index them all. Reports which projects are indexed vs remaining. Use autoIndex=true to batch-index all remaining projects sequentially.",
  {
    basePath: z
      .string()
      .describe("Absolute path to scan for project directories. Defaults to current working directory.")
      .optional(),
    autoIndex: z
      .boolean()
      .describe("When true, automatically index all remaining (non-indexed) projects sequentially.")
      .optional(),
    ignore: z
      .string()
      .describe("Comma-separated list of directory names to skip (e.g. 'KNIRV,n8n-master'). Default: 'KNIRV,n8n-master,node_modules'.")
      .optional(),
  },
  async (args) => ({
    content: [{ type: "text", text: await handleIndexTool("codebase_index_remaining", args) }],
  }),
);

// ── Query tools ──────────────────────────────────────────────────────────

srv.tool(
  "codebase_search",
  "Semantic search across an indexed codebase. Only use after codebase_index is complete (check codebase_status first). Returns relevant code chunks matching a natural language query.",
  {
    query: z.string().describe("Natural language search query (e.g. 'authentication middleware', 'database connection setup')."),
    projectPath: z
      .string()
      .describe("Absolute path to the project directory.")
      .optional(),
    limit: z
      .number()
      .min(1)
      .max(50)
      .describe("Maximum number of results to return. Default: 10 (override globally via SEARCH_DEFAULT_LIMIT env var).")
      .optional(),
    fileFilter: z
      .string()
      .describe("Filter results to a specific file path (relative).")
      .optional(),
    languageFilter: z
      .string()
      .describe("Filter results to a specific language (e.g. 'typescript', 'python').")
      .optional(),
    minScore: z
      .number()
      .min(0)
      .max(1)
      .describe("Minimum RRF score threshold (0-1). Results below this are filtered out. Default: 0.10 (override globally via SEARCH_MIN_SCORE env var). Set to 0 to disable filtering.")
      .optional(),
    includeLinked: z
      .boolean()
      .describe("When true, also search across linked projects defined in .socraticode.json or SOCRATICODE_LINKED_PROJECTS env var. Results include a project label showing which project each result came from. Default: false.")
      .optional(),
  },
  async (args) => ({
    content: [{ type: "text", text: await handleQueryTool("codebase_search", args) }],
  }),
);

srv.tool(
  "codebase_status",
  "Check index status: chunk count, indexing progress (%), last completed operation, file watcher state. Call after codebase_index to poll until 100% complete.",
  {
    projectPath: z
      .string()
      .describe("Absolute path to the project directory.")
      .optional(),
  },
  async (args) => ({
    content: [{ type: "text", text: await handleQueryTool("codebase_status", args) }],
  }),
);

// ── Graph tools ──────────────────────────────────────────────────────────

srv.tool(
  "codebase_graph_build",
  "Build a dependency graph of the codebase using static analysis (ast-grep). Maps import/require/export relationships between files. Runs in the background — call codebase_graph_status to poll progress until complete.",
  {
    projectPath: z
      .string()
      .describe("Absolute path to the project directory.")
      .optional(),
    extraExtensions: z
      .string()
      .describe("Comma-separated list of additional file extensions to include in the graph (e.g. '.tpl,.blade'). Files with non-standard extensions are included as leaf nodes (dependency targets). Can also be set globally via EXTRA_EXTENSIONS env var.")
      .optional(),
  },
  async (args) => ({
    content: [{ type: "text", text: await handleGraphTool("codebase_graph_build", args) }],
  }),
);

srv.tool(
  "codebase_graph_query",
  "Query the code dependency graph for a specific file. Returns what the file imports and what files depend on it.",
  {
    projectPath: z
      .string()
      .describe("Absolute path to the project directory.")
      .optional(),
    filePath: z.string().describe("Relative path of the file to query (e.g. 'src/index.ts')."),
  },
  async (args) => ({
    content: [{ type: "text", text: await handleGraphTool("codebase_graph_query", args) }],
  }),
);

srv.tool(
  "codebase_graph_stats",
  "Get statistics about the code dependency graph: total files, edges, most connected files, orphan files, circular dependencies.",
  {
    projectPath: z
      .string()
      .describe("Absolute path to the project directory.")
      .optional(),
  },
  async (args) => ({
    content: [{ type: "text", text: await handleGraphTool("codebase_graph_stats", args) }],
  }),
);

srv.tool(
  "codebase_graph_circular",
  "Find circular dependencies in the codebase.",
  {
    projectPath: z
      .string()
      .describe("Absolute path to the project directory.")
      .optional(),
  },
  async (args) => ({
    content: [{ type: "text", text: await handleGraphTool("codebase_graph_circular", args) }],
  }),
);

srv.tool(
  "codebase_graph_visualize",
  [
    "Visualise the code dependency graph. Two modes:",
    "  • mode=\"mermaid\" (default) — returns a Mermaid diagram (text) colour-coded by language, circular deps highlighted. Best for inline rendering inside chat, GitHub, or editors that render Mermaid.",
    "  • mode=\"interactive\" — writes a self-contained HTML page (vendored Cytoscape.js + Dagre, works offline) and opens it in the user's default browser. Shows the file graph and, when a symbol graph is available and fits, a Symbols toggle with the symbol-level call graph. Interactions: click node for sidebar with imports/dependents/symbols list; right-click node to highlight its blast radius (reverse-transitive closure); live search; layout switcher (Dagre / force / concentric / breadth-first / grid / circle); PNG export. Use this when the user asks for a visual/interactive view, wants to explore visually, or needs a shareable diagram.",
  ].join("\n"),
  {
    projectPath: z
      .string()
      .describe("Absolute path to the project directory.")
      .optional(),
    mode: z
      .enum(["mermaid", "interactive"])
      .describe("\"mermaid\" (default — text diagram) or \"interactive\" (browser-based explorer).")
      .optional(),
    open: z
      .boolean()
      .describe("In interactive mode, whether to auto-open the browser. Default true. Set false to just get the file path (useful in headless environments).")
      .optional(),
  },
  async (args) => ({
    content: [{ type: "text", text: await handleGraphTool("codebase_graph_visualize", args) }],
  }),
);

srv.tool(
  "codebase_graph_remove",
  "Remove a project's persisted code graph. Waits for any in-flight graph build to finish first. The graph can be rebuilt with codebase_graph_build or will be rebuilt automatically on the next codebase_index.",
  {
    projectPath: z.string().describe("Absolute path to the project directory."),
  },
  async (args) => ({
    content: [{ type: "text", text: await handleGraphTool("codebase_graph_remove", args) }],
  }),
);

srv.tool(
  "codebase_graph_status",
  "Check the status of the code dependency graph: build progress (if building), node/edge count, when it was last built, whether it's cached in memory. Use this to poll progress after calling codebase_graph_build.",
  {
    projectPath: z
      .string()
      .describe("Absolute path to the project directory.")
      .optional(),
  },
  async (args) => ({
    content: [{ type: "text", text: await handleGraphTool("codebase_graph_status", args) }],
  }),
);

// ── Impact analysis (symbol-level call graph) ───────────────────────────

srv.tool(
  "codebase_impact",
  "Impact Analysis — return the BLAST RADIUS for a file or symbol. Lists every file (and, where helpful, function) that could break if you change the target. Polymorphic on target: a path-like string ('src/foo.ts') triggers file-mode; a name-like string ('validateUser') triggers symbol-mode. Use this BEFORE refactoring, renaming, or deleting code to know what depends on it.",
  {
    projectPath: z.string().describe("Absolute path to the project directory.").optional(),
    target: z.string().describe("Target file path (relative) OR symbol name."),
    depth: z.number().describe("How many hops back to walk (default 3, max 10).").optional(),
  },
  async (args) => ({
    content: [{ type: "text", text: await handleGraphTool("codebase_impact", args) }],
  }),
);

srv.tool(
  "codebase_flow",
  "Trace the EXECUTION FLOW forward from an entry point — what does this code call into? With NO args, returns a ranked list of auto-detected entry points (orphans with outgoing calls, conventional names like main(), framework routes, tests). With an entrypoint argument, returns the call tree.",
  {
    projectPath: z.string().describe("Absolute path to the project directory.").optional(),
    entrypoint: z.string().describe("Symbol name to trace from. Omit to list auto-detected entry points.").optional(),
    file: z.string().describe("Optional file hint to disambiguate the symbol.").optional(),
    depth: z.number().describe("Maximum DFS depth (default 5, max 10).").optional(),
  },
  async (args) => ({
    content: [{ type: "text", text: await handleGraphTool("codebase_flow", args) }],
  }),
);

srv.tool(
  "codebase_symbol",
  "360° view of a symbol: definition, kind, callers, callees, confidence levels. Use to understand a function or class before changing it.",
  {
    projectPath: z.string().describe("Absolute path to the project directory.").optional(),
    name: z.string().describe("Symbol name (e.g. 'validateUser')."),
    file: z.string().describe("Optional file hint to disambiguate when the name is not unique.").optional(),
  },
  async (args) => ({
    content: [{ type: "text", text: await handleGraphTool("codebase_symbol", args) }],
  }),
);

srv.tool(
  "codebase_symbols",
  "List symbols in a file, or search by name across the project. Use to discover what exists before drilling into a single symbol with codebase_symbol.",
  {
    projectPath: z.string().describe("Absolute path to the project directory.").optional(),
    file: z.string().describe("Relative file path — list all symbols in this file.").optional(),
    query: z.string().describe("Substring to match against symbol names project-wide.").optional(),
    limit: z.number().describe("Maximum results (default 200).").optional(),
  },
  async (args) => ({
    content: [{ type: "text", text: await handleGraphTool("codebase_symbols", args) }],
  }),
);

// ── Context artifact tools ───────────────────────────────────────────────

srv.tool(
  "codebase_context",
  "List all context artifacts defined in .socraticodecontextartifacts.json — database schemas, API specs, infra configs, architecture docs, etc. Shows each artifact's name, description, path, and index status. Use this to discover what project knowledge is available beyond source code.",
  {
    projectPath: z
      .string()
      .describe("Absolute path to the project directory. If omitted, uses the current working directory.")
      .optional(),
  },
  async (args) => ({
    content: [{ type: "text", text: await handleContextTool("codebase_context", args) }],
  }),
);

srv.tool(
  "codebase_context_search",
  "Semantic search across context artifacts (database schemas, API specs, infra configs, etc.) defined in .socraticodecontextartifacts.json. Auto-indexes on first use and auto-detects stale artifacts. Use this to find relevant infrastructure or domain knowledge.",
  {
    query: z.string().describe("Natural language search query (e.g. 'tables related to billing', 'authentication endpoints', 'deployment resource limits')."),
    projectPath: z
      .string()
      .describe("Absolute path to the project directory.")
      .optional(),
    artifactName: z
      .string()
      .describe("Filter search to a specific artifact by name (e.g. 'database-schema'). Omit to search across all artifacts.")
      .optional(),
    limit: z
      .number()
      .min(1)
      .max(50)
      .describe("Maximum number of results to return. Default: 10.")
      .optional(),
    minScore: z
      .number()
      .min(0)
      .max(1)
      .describe("Minimum RRF score threshold (0-1). Results below this are filtered out. Default: 0.10 (override globally via SEARCH_MIN_SCORE env var). Set to 0 to disable filtering.")
      .optional(),
  },
  async (args) => ({
    content: [{ type: "text", text: await handleContextTool("codebase_context_search", args) }],
  }),
);

srv.tool(
  "codebase_context_index",
  "Index or re-index all context artifacts defined in .socraticodecontextartifacts.json. Chunks and embeds artifact content into the vector database for semantic search. Usually not needed — codebase_context_search auto-indexes on first use.",
  {
    projectPath: z
      .string()
      .describe("Absolute path to the project directory.")
      .optional(),
  },
  async (args) => ({
    content: [{ type: "text", text: await handleContextTool("codebase_context_index", args) }],
  }),
);

srv.tool(
  "codebase_context_remove",
  "Remove all indexed context artifacts for a project from the vector database. Blocked while indexing is in progress — use codebase_stop or wait for the operation to finish first.",
  {
    projectPath: z.string().describe("Absolute path to the project directory."),
  },
  async (args) => ({
    content: [{ type: "text", text: await handleContextTool("codebase_context_remove", args) }],
  }),
);

// ── Management tools ─────────────────────────────────────────────────────

srv.tool(
  "codebase_health",
  "Check the health of all infrastructure: Docker, Qdrant container, Ollama, and embedding model. Use this to diagnose setup issues.",
  {},
  async (args) => ({
    content: [{ type: "text", text: await handleManageTool("codebase_health", args) }],
  }),
);

srv.tool(
  "codebase_list_projects",
  "List all projects that have been indexed (have collections in Qdrant).",
  {},
  async (args) => ({
    content: [{ type: "text", text: await handleManageTool("codebase_list_projects", args) }],
  }),
);

srv.tool(
  "codebase_about",
  "Display information about SocratiCode — what it is, its tools and how to use it. Use this to get a quick overview of the MCP tools and their purpose.",
  {},
  async (args) => ({
    content: [{ type: "text", text: await handleManageTool("codebase_about", args) }],
  }),
);
}

registerAllTools(server);

// ── HTTP server session management ───────────────────────────────────────

interface McpSession {
  server: McpServer;
  transport: StreamableHTTPServerTransport;
}

const sessions = new Map<string, McpSession>();
let httpServer: ReturnType<typeof createServer> | null = null;

async function handleMcpRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
  try {
    const url = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);
    if (url.pathname !== "/mcp") {
      res.writeHead(404).end("Not found");
      return;
    }

    const sessionId = typeof req.headers["mcp-session-id"] === "string"
      ? req.headers["mcp-session-id"]
      : undefined;

    const session = sessionId ? sessions.get(sessionId) : undefined;

    if (req.method === "POST") {
      const buffers: Buffer[] = [];
      for await (const chunk of req) buffers.push(chunk);
      const body = JSON.parse(Buffer.concat(buffers).toString());

      if (!session) {
        const newServer = new McpServer(
          { name: "socraticode", version: SOCRATICODE_VERSION },
          { capabilities: { tools: {} } },
        );
        registerAllTools(newServer);

        const transport = new StreamableHTTPServerTransport({
          sessionIdGenerator: () => randomUUID(),
          onsessioninitialized: (sid) => {
            sessions.set(sid, { server: newServer, transport });
            logger.info("MCP session initialized", { sessionId: sid });
          },
          onsessionclosed: (sid) => {
            sessions.delete(sid);
            logger.info("MCP session closed", { sessionId: sid });
          },
        });

        transport.onclose = () => {
          const sid = transport.sessionId;
          if (sid) sessions.delete(sid);
        };

        await newServer.connect(transport);
        await transport.handleRequest(req, res, body);
        return;
      }

      await session.transport.handleRequest(req, res, body);
    } else if (req.method === "GET") {
      if (!session) {
        res.writeHead(400).end("Session ID required for GET");
        return;
      }
      await session.transport.handleRequest(req, res);
    } else if (req.method === "DELETE") {
      if (!session) {
        res.writeHead(400).end("Session ID required for DELETE");
        return;
      }
      await session.transport.handleRequest(req, res);
    } else {
      res.writeHead(405).end("Method not allowed");
    }
  } catch (err) {
    logger.error("HTTP request handler error", { error: String(err) });
    if (!res.headersSent) {
      res.writeHead(500).end("Internal server error");
    }
  }
}

async function startHttpServer(port: number): Promise<void> {
  const server = createServer(handleMcpRequest);
  httpServer = server;

  await new Promise<void>((resolve, reject) => {
    server.listen(port, () => {
      logger.info(`SocratiCode MCP server listening on http://localhost:${port}/mcp`);
      resolve();
    });
    server.once("error", reject);
  });
}

// ── Start server ─────────────────────────────────────────────────────────

async function main() {
  if (SOCRATICODE_PORT) {
    await startHttpServer(SOCRATICODE_PORT);
  } else {
    setMcpLogSender((params) => {
      server.server.sendLoggingMessage(params).catch(() => {});
    });
    const transport = new StdioServerTransport();
    await server.connect(transport);
  }

  // Auto-resume watchers and incremental updates for already-indexed projects
  // Fire-and-forget — runs in background, non-blocking, non-fatal
  autoResumeIndexedProjects();

  // ── Process-level error handlers ─────────────────────────────────────

  process.on("unhandledRejection", (reason) => {
    logger.error("Unhandled promise rejection", {
      error: reason instanceof Error ? reason.message : String(reason),
      stack: reason instanceof Error ? reason.stack : undefined,
    });
  });

  process.on("uncaughtException", (err) => {
    logger.error("Uncaught exception", {
      error: err.message,
      stack: err.stack,
    });
    // Uncaught exceptions leave the process in an undefined state — exit
    process.exit(1);
  });

  // ── Graceful shutdown ────────────────────────────────────────────────

  let shuttingDown = false;
  const shutdown = async (signal: string) => {
    if (shuttingDown) return; // prevent double shutdown
    shuttingDown = true;
    await gracefulShutdown(signal, async () => {
      if (httpServer) {
        const srv = httpServer;
        await new Promise<void>((resolve) => srv.close(() => resolve()));
      }
      // Close all active HTTP sessions
      for (const [, sess] of sessions) {
        try { await sess.transport.close(); } catch {}
      }
      await server.close();
    });
    process.exit(0);
  };

  process.on("SIGINT", () => shutdown("SIGINT"));
  process.on("SIGTERM", () => shutdown("SIGTERM"));

  // ── Stdin pipe-break detection (stdio mode only) ──────────────────────
  if (!SOCRATICODE_PORT) {
    process.stdin.on("end", () => shutdown("stdin EOF"));
    process.stdin.on("error", () => shutdown("stdin error"));
    process.stdin.on("close", () => shutdown("stdin close"));
  }
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
