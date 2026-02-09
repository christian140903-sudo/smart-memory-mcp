#!/usr/bin/env node

/**
 * Smart Memory MCP — Persistent Memory for AI Agents
 *
 * Give your AI agent a brain that never forgets.
 * TF-IDF semantic search, pattern recognition, proactive suggestions.
 * Works with Claude Code, Cursor, Windsurf, and any MCP client.
 *
 * https://nextool.app/memory
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { join } from "node:path";
import { existsSync } from "node:fs";
import { homedir } from "node:os";

// ═══════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════

const DB_NAME = process.env.SMART_MEMORY_DB || "default";
const DATA_DIR = join(
  process.env.SMART_MEMORY_DIR || join(homedir(), ".smart-memory"),
  DB_NAME
);
const KNOWLEDGE_FILE = join(DATA_DIR, "knowledge.json");
const INDEX_FILE = join(DATA_DIR, "index.json");

// ═══════════════════════════════════════════════════════════════
// TF-IDF ENGINE — Pure JS Semantic Similarity
// ═══════════════════════════════════════════════════════════════

const STOP_WORDS = new Set([
  "the","a","an","is","are","was","were","be","been","being","have","has",
  "had","do","does","did","will","would","could","should","may","might",
  "shall","can","need","to","of","in","for","on","with","at","by","from",
  "as","into","through","during","before","after","above","below","between",
  "out","off","over","under","again","further","then","once","here","there",
  "when","where","why","how","all","both","each","few","more","most","other",
  "some","such","no","nor","not","only","own","same","so","than","too",
  "very","just","because","but","and","or","if","while","that","this","it",
  "its","i","me","my","we","our","you","your","he","him","his","she","her",
  "they","them","their","what","which","who","whom","these","those","am",
  "about","up","das","die","der","und","ist","ein","eine","fuer","mit",
  "auf","von","den","dem","des","ich","du","er","sie","es","wir","ihr",
]);

function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\u00e4\u00f6\u00fc\u00df\s-]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 2 && !STOP_WORDS.has(w));
}

function termFrequency(tokens) {
  const tf = {};
  for (const t of tokens) tf[t] = (tf[t] || 0) + 1;
  const max = Math.max(...Object.values(tf), 1);
  for (const t in tf) tf[t] = tf[t] / max;
  return tf;
}

function cosineSimilarity(a, b) {
  const terms = new Set([...Object.keys(a), ...Object.keys(b)]);
  let dot = 0, magA = 0, magB = 0;
  for (const t of terms) {
    const va = a[t] || 0, vb = b[t] || 0;
    dot += va * vb;
    magA += va * va;
    magB += vb * vb;
  }
  return magA && magB ? dot / (Math.sqrt(magA) * Math.sqrt(magB)) : 0;
}

// ═══════════════════════════════════════════════════════════════
// STORAGE
// ═══════════════════════════════════════════════════════════════

async function ensureDir() {
  if (!existsSync(DATA_DIR)) await mkdir(DATA_DIR, { recursive: true });
}

async function loadJson(path, fallback) {
  try {
    if (!existsSync(path)) return fallback;
    return JSON.parse(await readFile(path, "utf-8"));
  } catch {
    return fallback;
  }
}

async function saveJson(path, data) {
  await ensureDir();
  await writeFile(path, JSON.stringify(data, null, 2));
}

async function loadKB() {
  return loadJson(KNOWLEDGE_FILE, { entries: [], nextId: 1 });
}

async function saveKB(kb) {
  await saveJson(KNOWLEDGE_FILE, kb);
}

function buildIndex(entries) {
  const terms = {};
  const n = entries.length;
  for (const entry of entries) {
    const tokens = new Set(
      tokenize(`${entry.content} ${(entry.tags || []).join(" ")} ${entry.category || ""}`)
    );
    for (const token of tokens) {
      if (!terms[token]) terms[token] = [];
      terms[token].push(entry.id);
    }
  }
  const idf = {};
  for (const [term, docs] of Object.entries(terms)) {
    idf[term] = Math.log((n + 1) / (docs.length + 1)) + 1;
  }
  return { terms, idf, docCount: n };
}

// ═══════════════════════════════════════════════════════════════
// MEMORY OPERATIONS
// ═══════════════════════════════════════════════════════════════

async function memoryLearn({ content, category, tags, source }) {
  const kb = await loadKB();
  const entry = {
    id: kb.nextId++,
    content,
    category: category || "general",
    tags: tags || [],
    source: source || "manual",
    timestamp: new Date().toISOString(),
    accessCount: 0,
    usefulness: 0,
  };
  kb.entries.push(entry);
  await saveKB(kb);
  const idx = buildIndex(kb.entries);
  await saveJson(INDEX_FILE, idx);
  return {
    id: entry.id,
    total: kb.entries.length,
    message: `Stored: "${content.substring(0, 100)}${content.length > 100 ? "..." : ""}" [${entry.category}]`,
  };
}

async function memoryRecall({ query, limit, category, minScore }) {
  limit = limit || 10;
  minScore = minScore || 0.1;
  const kb = await loadKB();
  if (!kb.entries.length) return { results: [], message: "Knowledge base is empty. Use memory_learn to add knowledge." };

  const idx = await loadJson(INDEX_FILE, buildIndex(kb.entries));
  const queryTokens = tokenize(query);
  const queryTf = termFrequency(queryTokens);
  const queryVec = {};
  for (const [term, tf] of Object.entries(queryTf)) {
    queryVec[term] = tf * (idx.idf[term] || 1);
  }

  const scored = kb.entries
    .filter((e) => !category || e.category === category)
    .map((entry) => {
      const tokens = tokenize(`${entry.content} ${(entry.tags || []).join(" ")} ${entry.category || ""}`);
      const entryTf = termFrequency(tokens);
      const entryVec = {};
      for (const [term, tf] of Object.entries(entryTf)) {
        entryVec[term] = tf * (idx.idf[term] || 1);
      }
      let score = cosineSimilarity(queryVec, entryVec);
      // Recency boost (last 24h: +10%)
      const age = Date.now() - new Date(entry.timestamp).getTime();
      if (age < 86400000) score += 0.1;
      // Usefulness boost
      score += entry.usefulness * 0.05;
      return { entry, score };
    })
    .filter((s) => s.score >= minScore)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);

  // Update access counts
  for (const s of scored) {
    const e = kb.entries.find((e) => e.id === s.entry.id);
    if (e) e.accessCount++;
  }
  await saveKB(kb);

  return {
    results: scored.map((s) => ({
      id: s.entry.id,
      content: s.entry.content,
      category: s.entry.category,
      tags: s.entry.tags,
      score: Math.round(s.score * 1000) / 1000,
      timestamp: s.entry.timestamp,
    })),
    total: kb.entries.length,
  };
}

async function memoryStats() {
  const kb = await loadKB();
  const categories = {};
  let totalAccess = 0;
  let totalUsefulness = 0;
  for (const e of kb.entries) {
    categories[e.category] = (categories[e.category] || 0) + 1;
    totalAccess += e.accessCount;
    totalUsefulness += e.usefulness;
  }
  return {
    totalEntries: kb.entries.length,
    categories,
    totalAccesses: totalAccess,
    avgUsefulness: kb.entries.length ? Math.round((totalUsefulness / kb.entries.length) * 100) / 100 : 0,
    oldestEntry: kb.entries[0]?.timestamp || null,
    newestEntry: kb.entries[kb.entries.length - 1]?.timestamp || null,
    storagePath: DATA_DIR,
    database: DB_NAME,
  };
}

async function memoryPatterns() {
  const kb = await loadKB();
  if (kb.entries.length < 3) return { message: "Need at least 3 entries for pattern detection." };

  const categories = {};
  const tagFreq = {};
  const coOccurrence = {};

  for (const entry of kb.entries) {
    categories[entry.category] = (categories[entry.category] || 0) + 1;
    for (const tag of entry.tags || []) tagFreq[tag] = (tagFreq[tag] || 0) + 1;
    const tags = entry.tags || [];
    for (let i = 0; i < tags.length; i++) {
      for (let j = i + 1; j < tags.length; j++) {
        const pair = [tags[i], tags[j]].sort().join(" + ");
        coOccurrence[pair] = (coOccurrence[pair] || 0) + 1;
      }
    }
  }

  const mostAccessed = [...kb.entries]
    .sort((a, b) => b.accessCount - a.accessCount)
    .slice(0, 5)
    .map((e) => ({ id: e.id, content: e.content.substring(0, 120), accessCount: e.accessCount }));

  return {
    totalEntries: kb.entries.length,
    categories: Object.entries(categories).sort((a, b) => b[1] - a[1]),
    topTags: Object.entries(tagFreq).sort((a, b) => b[1] - a[1]).slice(0, 10),
    tagPairs: Object.entries(coOccurrence).sort((a, b) => b[1] - a[1]).slice(0, 5),
    mostAccessed,
  };
}

async function memorySuggest({ context }) {
  const kb = await loadKB();
  if (!kb.entries.length) return { suggestions: [], message: "No knowledge yet." };

  const recall = await memoryRecall({ query: context, limit: 5, minScore: 0.05 });
  const suggestions = recall.results.map((r) => {
    const type = (r.tags || []).some((t) => ["error", "failure"].includes(t))
      ? "warning"
      : (r.tags || []).some((t) => ["success", "solution"].includes(t))
        ? "recommendation"
        : (r.tags || []).some((t) => ["pattern", "best-practice"].includes(t))
          ? "best_practice"
          : "related";
    return { type, content: r.content.substring(0, 200), source: r.id, relevance: r.score };
  });

  return { suggestions, context: context.substring(0, 80) };
}

async function memoryEvaluate({ entryId, useful, feedback }) {
  const kb = await loadKB();
  const entry = kb.entries.find((e) => e.id === entryId);
  if (!entry) return { success: false, message: `Entry ${entryId} not found.` };
  entry.usefulness += useful ? 1 : -1;
  if (feedback) {
    entry.feedback = entry.feedback || [];
    entry.feedback.push({ text: feedback, timestamp: new Date().toISOString() });
  }
  await saveKB(kb);
  return { success: true, entryId, usefulness: entry.usefulness };
}

// ═══════════════════════════════════════════════════════════════
// MCP SERVER
// ═══════════════════════════════════════════════════════════════

const server = new Server(
  { name: "smart-memory-mcp", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

const TOOLS = [
  {
    name: "memory_learn",
    description:
      "Store new knowledge in persistent memory. Knowledge persists across sessions and is searchable via semantic similarity. Use this to remember important context, decisions, patterns, solutions, and anything your agent should recall later.",
    inputSchema: {
      type: "object",
      properties: {
        content: { type: "string", description: "The knowledge to store. Be specific and descriptive for better recall." },
        category: { type: "string", description: "Category for organization (e.g., 'error', 'solution', 'pattern', 'project', 'preference')" },
        tags: { type: "array", items: { type: "string" }, description: "Tags for retrieval (e.g., ['typescript', 'debugging', 'success'])" },
        source: { type: "string", description: "Where this knowledge came from (e.g., 'user', 'codebase', 'web')" },
      },
      required: ["content"],
    },
  },
  {
    name: "memory_recall",
    description:
      "Search persistent memory using TF-IDF semantic similarity. Returns the most relevant knowledge entries ranked by relevance score. Use this to recall context, solutions, patterns, and decisions from previous sessions.",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "What to search for. Natural language works best." },
        limit: { type: "number", description: "Max results to return (default: 10)" },
        category: { type: "string", description: "Filter by category (optional)" },
        minScore: { type: "number", description: "Minimum relevance score 0-1 (default: 0.1)" },
      },
      required: ["query"],
    },
  },
  {
    name: "memory_stats",
    description:
      "Get statistics about the knowledge base: total entries, categories, access patterns, and storage info.",
    inputSchema: { type: "object", properties: {} },
  },
  {
    name: "memory_patterns",
    description:
      "Analyze patterns in stored knowledge: category distribution, tag frequency, co-occurring tags, most accessed entries. Requires at least 3 entries.",
    inputSchema: { type: "object", properties: {} },
  },
  {
    name: "memory_suggest",
    description:
      "Get proactive suggestions based on current context. Returns warnings about past errors, successful approaches, and best practices relevant to what you're working on.",
    inputSchema: {
      type: "object",
      properties: {
        context: { type: "string", description: "Current task or context to get suggestions for" },
      },
      required: ["context"],
    },
  },
  {
    name: "memory_evaluate",
    description:
      "Rate a knowledge entry as useful or not useful. Improves future recall quality by boosting useful entries.",
    inputSchema: {
      type: "object",
      properties: {
        entryId: { type: "number", description: "ID of the knowledge entry to rate" },
        useful: { type: "boolean", description: "Whether the entry was useful" },
        feedback: { type: "string", description: "Optional feedback text" },
      },
      required: ["entryId", "useful"],
    },
  },
];

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: TOOLS,
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  try {
    let result;
    switch (name) {
      case "memory_learn":
        result = await memoryLearn(args);
        break;
      case "memory_recall":
        result = await memoryRecall(args);
        break;
      case "memory_stats":
        result = await memoryStats();
        break;
      case "memory_patterns":
        result = await memoryPatterns();
        break;
      case "memory_suggest":
        result = await memorySuggest(args);
        break;
      case "memory_evaluate":
        result = await memoryEvaluate(args);
        break;
      default:
        return { content: [{ type: "text", text: `Unknown tool: ${name}` }], isError: true };
    }
    return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
  } catch (err) {
    return { content: [{ type: "text", text: `Error: ${err.message}` }], isError: true };
  }
});

// ═══════════════════════════════════════════════════════════════
// START
// ═══════════════════════════════════════════════════════════════

async function main() {
  await ensureDir();
  const transport = new StdioServerTransport();
  await server.connect(transport);
  process.stderr.write(`Smart Memory MCP v1.0.0 | DB: ${DB_NAME} | Path: ${DATA_DIR}\n`);
}

main().catch((err) => {
  process.stderr.write(`Fatal: ${err.message}\n`);
  process.exit(1);
});
