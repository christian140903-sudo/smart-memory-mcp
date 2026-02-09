# Smart Memory MCP

**Persistent memory with semantic search for AI agents.**

Your AI agent forgets everything between sessions. Smart Memory fixes that.

Works with **Claude Code**, **Cursor**, **Windsurf**, and any MCP-compatible client.

## Features

- **Semantic Search** — TF-IDF similarity matching, not just keywords
- **Zero Config** — Install and it works. No cloud, no API keys
- **100% Local** — All data stays on your machine as JSON files
- **Bilingual** — English and German language support
- **Recency Boost** — Recent knowledge ranks higher
- **Usefulness Scoring** — Rate entries to improve future recall

## Quick Start

### Claude Code

```bash
claude mcp add memory -- npx mcp-smart-memory
```

### Cursor / Windsurf / Other MCP Clients

Add to your MCP configuration:

```json
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["mcp-smart-memory"]
    }
  }
}
```

That's it. Your agent now has persistent memory.

## Tools

| Tool | Description |
|------|-------------|
| `memory_learn` | Store knowledge with categories and tags |
| `memory_recall` | Search using TF-IDF semantic similarity |
| `memory_stats` | Knowledge base statistics and overview |
| `memory_patterns` | Analyze patterns, tag frequency, co-occurrence |
| `memory_suggest` | Proactive suggestions based on current context |
| `memory_evaluate` | Rate entries as useful/not useful for better recall |

## Usage Examples

### Store knowledge
```
memory_learn: "React useEffect cleanup runs before component unmount and before re-running the effect"
  category: "pattern"
  tags: ["react", "hooks", "cleanup"]
```

### Recall knowledge
```
memory_recall: "how to clean up effects in React"
→ Returns semantically similar entries ranked by relevance
```

### Get suggestions
```
memory_suggest: "I'm debugging a memory leak in a React component"
→ Returns relevant warnings, solutions, and best practices from your knowledge base
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `SMART_MEMORY_DIR` | `~/.smart-memory` | Data storage directory |
| `SMART_MEMORY_DB` | `default` | Database name (for multiple knowledge bases) |

### Multiple Knowledge Bases

```json
{
  "mcpServers": {
    "work-memory": {
      "command": "npx",
      "args": ["mcp-smart-memory"],
      "env": { "SMART_MEMORY_DB": "work" }
    },
    "personal-memory": {
      "command": "npx",
      "args": ["mcp-smart-memory"],
      "env": { "SMART_MEMORY_DB": "personal" }
    }
  }
}
```

## How It Works

Smart Memory uses **TF-IDF (Term Frequency-Inverse Document Frequency)** with cosine similarity for semantic search — implemented in pure JavaScript with zero ML dependencies.

When you store knowledge:
1. Text is tokenized and stop-words are removed
2. An inverted index is built for fast lookup
3. TF-IDF vectors are computed for similarity matching

When you search:
1. Your query is vectorized using the same TF-IDF pipeline
2. Cosine similarity is computed against all entries
3. Results are boosted by recency and usefulness ratings
4. Top matches are returned, ranked by relevance score

## Data Storage

All data is stored locally as JSON files:

```
~/.smart-memory/
  └── default/
      ├── knowledge.json    # Your knowledge entries
      └── index.json        # TF-IDF search index
```

Files are human-readable and portable. Back them up, share them, or version control them.

## Why Smart Memory?

| | Smart Memory | claude-mem | mem0 | Zep |
|---|---|---|---|---|
| MCP Native | ✅ | ❌ Plugin | ❌ SDK | ❌ SDK |
| Any MCP Client | ✅ | ❌ Claude only | ❌ Python only | ❌ Python only |
| No Cloud | ✅ | ✅ | ❌ | ❌ |
| No API Key | ✅ | ✅ | ❌ | ❌ |
| Semantic Search | ✅ TF-IDF | ❌ Basic | ✅ Embeddings | ✅ Embeddings |
| Pattern Detection | ✅ | ❌ | ❌ | ❌ |
| Proactive Suggestions | ✅ | ❌ | ❌ | ❌ |

## Built With

Built by [NexTool](https://nextool.app) — an AI-native development studio powered by Claude Opus 4.6.

## License

MIT
