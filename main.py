import json
import time
from pathlib import Path

import httpx
from mcp.server import MCPServer

HOSTS_FILE = Path(__file__).parent / "hosts.json"
TIMEOUT = 10.0

mcp = MCPServer("ollama-fleet")


def _load_hosts() -> dict[str, str]:
    return json.loads(HOSTS_FILE.read_text())


def _resolve_host(name: str) -> str:
    hosts = _load_hosts()
    if name not in hosts:
        raise ValueError(f"unknown host '{name}'. known hosts: {', '.join(hosts)}")
    return hosts[name]


@mcp.tool()
async def list_models(host: str | None = None) -> dict:
    """List installed Ollama models per fleet host.

    Args:
        host: optional host name (e.g. "corsair"). If omitted, lists all fleet hosts.
    """
    hosts = _load_hosts()
    targets = {host: hosts[host]} if host else hosts
    if host and host not in hosts:
        return {"error": f"unknown host '{host}'. known hosts: {', '.join(hosts)}"}

    result: dict[str, object] = {}
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for name, base_url in targets.items():
            try:
                resp = await client.get(f"{base_url}/api/tags")
                resp.raise_for_status()
                models = [m["name"] for m in resp.json().get("models", [])]
                result[name] = models
            except Exception as exc:
                result[name] = {"error": str(exc)}
    return result


@mcp.tool()
async def generate(host: str, model: str, prompt: str, timeout_sec: float = 120.0) -> dict:
    """Send a prompt to a specific model on a specific fleet host and return its response.

    Args:
        host: fleet host name (e.g. "mac-studio", "corsair", "alien18").
        model: model name as reported by list_models (e.g. "qwen3:32b").
        prompt: the prompt text to send.
        timeout_sec: max seconds to wait for a response (default 120).
    """
    try:
        base_url = _resolve_host(host)
    except ValueError as exc:
        return {"error": str(exc)}

    async with httpx.AsyncClient(timeout=timeout_sec) as client:
        try:
            start = time.monotonic()
            resp = await client.post(
                f"{base_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = time.monotonic() - start
            return {
                "host": host,
                "model": model,
                "response": data.get("response", ""),
                "elapsed_sec": round(elapsed, 2),
            }
        except Exception as exc:
            return {"host": host, "model": model, "error": str(exc)}


@mcp.tool()
async def fleet_health() -> dict:
    """Check reachability and model count for every host in the fleet."""
    hosts = _load_hosts()
    result: dict[str, object] = {}
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for name, base_url in hosts.items():
            start = time.monotonic()
            try:
                resp = await client.get(f"{base_url}/api/tags")
                resp.raise_for_status()
                elapsed_ms = round((time.monotonic() - start) * 1000)
                model_count = len(resp.json().get("models", []))
                result[name] = {"status": "up", "latency_ms": elapsed_ms, "models": model_count}
            except Exception as exc:
                result[name] = {"status": "down", "error": str(exc)}
    return result


if __name__ == "__main__":
    mcp.run()
