from __future__ import annotations

import shutil
import subprocess
import time

import httpx

from kepler.engines.base import (
    EngineCapabilities,
    EngineResult,
    InferenceEngine,
    ModelFormat,
)
from kepler.models.registry import ModelSpec

OLLAMA_BASE_URL = "http://localhost:11434"
HEALTH_TIMEOUT_S = 1.0
PULL_TIMEOUT_S = 600.0
GENERATE_TIMEOUT_S = 600.0
DAEMON_READY_TIMEOUT_S = 30.0


class OllamaEngine(InferenceEngine):
    """Ollama is a user-managed daemon. We discover or start it, pull the tag if
    missing, run inference via REST, then unload (don't kill the daemon)."""

    name = "ollama"

    def __init__(self, base_url: str = OLLAMA_BASE_URL, owns_daemon: bool = False):
        self.base_url = base_url
        self._owns_daemon = owns_daemon
        self._daemon_proc: subprocess.Popen | None = None
        self._tag: str | None = None
        self._client = httpx.Client(timeout=GENERATE_TIMEOUT_S)

    @property
    def capabilities(self) -> EngineCapabilities:
        avail, reason = _probe()
        return EngineCapabilities(
            name=self.name,
            supported_formats={ModelFormat.GGUF, ModelFormat.MLX},
            available=avail,
            unavailable_reason=reason,
            requires_server=True,
        )

    def load(self, model: ModelSpec) -> None:
        if model.ollama_tag is None:
            raise RuntimeError("OllamaEngine.load: ModelSpec.ollama_tag is required")
        self._tag = model.ollama_tag
        if not _daemon_reachable(self.base_url):
            self._spawn_daemon()
        if not _tag_present(self._client, self.base_url, self._tag):
            self._pull(self._tag)

    def infer(self, prompt: str, max_tokens: int, temperature: float) -> EngineResult:
        if self._tag is None:
            raise RuntimeError("OllamaEngine.infer called before load()")
        payload = {
            "model": self._tag,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        t0 = time.perf_counter()
        resp = self._client.post(f"{self.base_url}/api/generate", json=payload)
        resp.raise_for_status()
        wall_s = time.perf_counter() - t0
        data = resp.json()

        text = data.get("response", "")
        prompt_tokens = int(data.get("prompt_eval_count", 0))
        generated_tokens = int(data.get("eval_count", 0))
        prefill_ms = _ns_to_ms(data.get("prompt_eval_duration", 0))
        generation_ms = _ns_to_ms(data.get("eval_duration", 0))
        gen_tps = generated_tokens / (generation_ms / 1000.0) if generation_ms > 0 else 0.0
        tok_per_s = generated_tokens / wall_s if wall_s > 0 else 0.0

        return EngineResult(
            output_text=text,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            wall_s=wall_s,
            ttft_ms=prefill_ms,
            prefill_ms=prefill_ms,
            generation_ms=generation_ms,
            generation_tok_per_s=gen_tps,
            tok_per_s=tok_per_s,
        )

    def unload(self) -> None:
        if self._tag is not None:
            try:
                self._client.post(
                    f"{self.base_url}/api/generate",
                    json={"model": self._tag, "keep_alive": 0, "prompt": ""},
                    timeout=10.0,
                )
            except Exception:
                pass
        try:
            self._client.close()
        except Exception:
            pass
        if self._owns_daemon and self._daemon_proc is not None:
            try:
                self._daemon_proc.terminate()
                try:
                    self._daemon_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._daemon_proc.kill()
                    self._daemon_proc.wait(timeout=5)
            except Exception:
                pass
            self._daemon_proc = None
        self._tag = None

    def _spawn_daemon(self) -> None:
        if shutil.which("ollama") is None:
            raise RuntimeError("Ollama daemon not running and `ollama` binary not on PATH")
        self._daemon_proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._owns_daemon = True
        deadline = time.monotonic() + DAEMON_READY_TIMEOUT_S
        while time.monotonic() < deadline:
            if _daemon_reachable(self.base_url):
                return
            time.sleep(0.5)
        raise RuntimeError(f"Ollama daemon failed to become ready within {DAEMON_READY_TIMEOUT_S}s")

    def _pull(self, tag: str) -> None:
        with self._client.stream(
            "POST",
            f"{self.base_url}/api/pull",
            json={"name": tag, "stream": True},
            timeout=PULL_TIMEOUT_S,
        ) as resp:
            resp.raise_for_status()
            for _ in resp.iter_lines():
                pass


def _ns_to_ms(ns: int | float | None) -> float:
    if not ns:
        return 0.0
    return float(ns) / 1_000_000.0


def _daemon_reachable(base_url: str) -> bool:
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=HEALTH_TIMEOUT_S)
        return r.status_code == 200
    except Exception:
        return False


def _tag_present(client: httpx.Client, base_url: str, tag: str) -> bool:
    try:
        r = client.get(f"{base_url}/api/tags", timeout=5.0)
        if r.status_code != 200:
            return False
        names = {m.get("name") for m in (r.json().get("models") or [])}
        return tag in names
    except Exception:
        return False


def _probe() -> tuple[bool, str | None]:
    if shutil.which("ollama") is not None:
        return True, None
    if _daemon_reachable(OLLAMA_BASE_URL):
        return True, None
    return False, "Ollama not installed (binary not on PATH and daemon not reachable)"


def derive_tag_from_repo(repo_id: str) -> str:
    """Best-effort heuristic to map an HF repo to an Ollama tag.

    Examples:
        Qwen/Qwen2.5-0.5B-Instruct -> qwen2.5:0.5b
        meta-llama/Llama-3.2-1B    -> llama3.2:1b
        google/gemma-2-2b-it       -> gemma2:2b
    Returns lowercase. Override with --ollama-tag for accuracy.
    """
    name = repo_id.split("/", 1)[-1].lower()
    parts = name.replace("_", "-").split("-")
    base = parts[0]
    size = ""
    for p in parts[1:]:
        if p and p[0].isdigit() and any(p.endswith(s) for s in ("b", "m")):
            size = p
            break
        if p.endswith("b") and p[:-1].replace(".", "").isdigit():
            size = p
            break
    if not size:
        size = "latest"
    return f"{base}:{size}"
