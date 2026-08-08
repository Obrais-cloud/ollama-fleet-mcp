# ollama-fleet-mcp

MCP server that exposes the local Ollama fleet (mac-studio, corsair, alien18) as tools for Claude Code.

Rewritten 2026-08-08 — the original was lost with the X10 Pro_A SSD; no backup or GitHub remote existed. This is a fresh implementation covering the same purpose (fleet-aware Ollama routing/inspection from inside Claude Code), not a recovery of the original code.

## Tools

- `list_models(host=None)` — installed models per host, or all hosts if omitted.
- `generate(host, model, prompt, timeout_sec=120)` — send a prompt to a specific host+model, return the response.
- `fleet_health()` — reachability, latency, and model count for every host.
- `compare_models(prompt, targets)` — send the same prompt to multiple `{host, model}` pairs in parallel, return responses side by side.
- `smart_generate(model, prompt, timeout_sec=120)` — routes to whichever host already has `model` loaded in memory (checks `/api/ps`), falling back to any host that has it installed. Avoids cold-load latency.
- `pull_model(host, model, timeout_sec=900)` — pull/verify a model on a specific host.

## Hosts

Configured in `hosts.json` (Tailscale IPs):

```json
{
  "mac-studio": "http://100.68.94.14:11434",
  "corsair": "http://100.94.117.48:11434",
  "alien18": "http://100.87.2.47:11434"
}
```

mac mini is intentionally excluded — its local Ollama is loopback-only, used internally by ollaroute/ollafifo/ollasecret, not a fleet compute node.

## Run

```
uv run python main.py
```

## Register with Claude Code

```
claude mcp add ollama-fleet -- uv run --directory ~/ollama-fleet-mcp python main.py
```
