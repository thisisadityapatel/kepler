from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

from kepler.engines.base import (
    EngineCapabilities,
    EngineResult,
    InferenceEngine,
    ModelFormat,
)
from kepler.models.registry import ModelSpec

LLAMACPP_REPO_URL = "https://github.com/ggml-org/llama.cpp.git"
SERVER_READY_TIMEOUT_S = 60.0
GENERATE_TIMEOUT_S = 600.0
BUILD_TIMEOUT_S = 1800.0
HEALTH_TIMEOUT_S = 1.0


class LlamaCppBinaryEngine(InferenceEngine):
    """GGUF inference via the upstream llama.cpp C++ build. Clones the repo and
    builds `llama-server` once into `inference_engines/llama.cpp/build/`; the build
    is persistent so subsequent runs reuse it. Each load() spawns a server, each
    infer() POSTs to /completion, unload() shuts the server down."""

    name = "llamacpp-binary"

    def __init__(self, n_ctx: int = 4096, n_gpu_layers: int = 999):
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self._port: int | None = None
        self._proc: subprocess.Popen | None = None
        self._client: httpx.Client | None = None
        self._spec: ModelSpec | None = None
        self._stderr_path: Path | None = None
        self._stderr_file = None

    @property
    def capabilities(self) -> EngineCapabilities:
        avail, reason = _probe()
        return EngineCapabilities(
            name=self.name,
            supported_formats={ModelFormat.GGUF},
            available=avail,
            unavailable_reason=reason,
            requires_server=True,
        )

    def load(self, model: ModelSpec) -> None:
        if self._proc is not None:
            return
        if model.local_path is None:
            raise RuntimeError("LlamaCppBinaryEngine.load: ModelSpec.local_path is required")

        binary = _ensure_built()
        self._port = _pick_free_port()
        # Pipe stderr to a temp file so (a) the pipe buffer can't fill and deadlock
        # the server during long runs, and (b) we can surface the actual error if
        # the server exits before becoming ready.
        self._stderr_file = tempfile.NamedTemporaryFile(
            prefix="kepler-llamacpp-binary-", suffix=".stderr.log", delete=False
        )
        self._stderr_path = Path(self._stderr_file.name)
        self._proc = subprocess.Popen(
            [
                str(binary),
                "-m", str(model.local_path),
                "--port", str(self._port),
                "--host", "127.0.0.1",
                "-c", str(self.n_ctx),
                "-ngl", str(self.n_gpu_layers),
                "--flash-attn", "on",
                "--no-ui",
            ],
            stdout=subprocess.DEVNULL,
            stderr=self._stderr_file,
        )
        base_url = f"http://127.0.0.1:{self._port}"
        self._client = httpx.Client(base_url=base_url, timeout=GENERATE_TIMEOUT_S)
        try:
            _wait_for_server(base_url, self._proc, self._stderr_path)
        except Exception:
            self.unload()
            raise
        self._spec = model

    def infer(self, prompt: str, max_tokens: int, temperature: float) -> EngineResult:
        if self._client is None:
            raise RuntimeError("LlamaCppBinaryEngine.infer called before load()")

        t0 = time.perf_counter()
        resp = self._client.post(
            "/completion",
            json={
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": temperature,
                "stream": False,
                "cache_prompt": False,
            },
        )
        resp.raise_for_status()
        wall_s = time.perf_counter() - t0
        data = resp.json()

        text = data.get("content", "")
        timings = data.get("timings", {}) or {}
        prompt_tokens = int(timings.get("prompt_n", 0) or 0)
        generated_tokens = int(timings.get("predicted_n", 0) or 0)
        prefill_ms = float(timings.get("prompt_ms", 0.0) or 0.0)
        generation_ms = float(timings.get("predicted_ms", 0.0) or 0.0)
        gen_tps = (
            float(timings.get("predicted_per_second", 0.0) or 0.0)
            if timings.get("predicted_per_second") is not None
            else (generated_tokens / (generation_ms / 1000.0) if generation_ms > 0 else 0.0)
        )
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
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        if self._proc is not None:
            try:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=5)
            except Exception:
                pass
            self._proc = None
        if self._stderr_file is not None:
            try:
                self._stderr_file.close()
            except Exception:
                pass
            self._stderr_file = None
        if self._stderr_path is not None:
            try:
                self._stderr_path.unlink(missing_ok=True)
            except Exception:
                pass
            self._stderr_path = None
        self._port = None
        self._spec = None


def _project_root() -> Path:
    """Locate the kepler project root by walking up from CWD looking for the
    config/ folder (same heuristic as kepler.config.loader.find_config)."""
    here = Path.cwd().resolve()
    for parent in [here, *here.parents]:
        if (parent / "config" / "models.yaml").is_file():
            return parent
    # Fall back to walking up from this source file — works when running
    # from a wheel installed in editable mode.
    src = Path(__file__).resolve()
    for parent in src.parents:
        if (parent / "config" / "models.yaml").is_file():
            return parent
    raise FileNotFoundError("kepler project root (containing config/models.yaml) not found")


def _llamacpp_dir() -> Path:
    return _project_root() / "inference_engines" / "llama.cpp"


def _binary_path() -> Path:
    return _llamacpp_dir() / "build" / "bin" / "llama-server"


def _ensure_built() -> Path:
    """Clone llama.cpp and build llama-server if the binary isn't already on disk.
    The build is cached under inference_engines/llama.cpp/ and persists across runs."""
    binary = _binary_path()
    if binary.is_file() and os.access(binary, os.X_OK):
        return binary

    src_dir = _llamacpp_dir()
    src_dir.parent.mkdir(parents=True, exist_ok=True)

    if not (src_dir / ".git").is_dir():
        print(
            f"[llamacpp-binary] cloning {LLAMACPP_REPO_URL} into {src_dir} …",
            file=sys.stderr,
            flush=True,
        )
        subprocess.run(
            ["git", "clone", "--depth", "1", LLAMACPP_REPO_URL, str(src_dir)],
            check=True,
            timeout=BUILD_TIMEOUT_S,
        )

    build_dir = src_dir / "build"
    print(
        f"[llamacpp-binary] configuring CMake build (Metal=ON) in {build_dir} …",
        file=sys.stderr,
        flush=True,
    )
    cmake_args = [
        "cmake", "-S", str(src_dir), "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_BUILD_SERVER=ON",
        "-DLLAMA_CURL=OFF",
    ]
    if platform.system() == "Darwin":
        cmake_args.append("-DGGML_METAL=ON")
    subprocess.run(cmake_args, check=True, timeout=BUILD_TIMEOUT_S)

    print("[llamacpp-binary] building llama-server …", file=sys.stderr, flush=True)
    subprocess.run(
        [
            "cmake", "--build", str(build_dir),
            "--config", "Release",
            "--target", "llama-server",
            "-j",
        ],
        check=True,
        timeout=BUILD_TIMEOUT_S,
    )

    if not (binary.is_file() and os.access(binary, os.X_OK)):
        raise RuntimeError(
            f"llama-server build finished but {binary} is missing — "
            f"inspect the build output in {build_dir}"
        )
    print(f"[llamacpp-binary] built {binary}", file=sys.stderr, flush=True)
    return binary


def _pick_free_port() -> int:
    """Ask the kernel for an unused port. Closes the socket immediately; there is
    a brief race window before llama-server binds, which is fine for a benchmark."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(base_url: str, proc: subprocess.Popen, stderr_path: Path) -> None:
    deadline = time.monotonic() + SERVER_READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = _tail_stderr(stderr_path)
            raise RuntimeError(
                f"llama-server exited with code {proc.returncode} before becoming ready"
                + (f"\n--- stderr (tail) ---\n{tail}" if tail else "")
            )
        try:
            r = httpx.get(f"{base_url}/health", timeout=HEALTH_TIMEOUT_S)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.25)
    raise RuntimeError(f"llama-server did not become ready within {SERVER_READY_TIMEOUT_S}s")


def _tail_stderr(path: Path, max_chars: int = 2000) -> str:
    try:
        data = path.read_text(errors="replace")
    except Exception:
        return ""
    if len(data) > max_chars:
        return "…" + data[-max_chars:]
    return data


def _probe() -> tuple[bool, str | None]:
    if _binary_path().is_file():
        return True, None
    missing: list[str] = []
    for tool in ("git", "cmake"):
        if shutil.which(tool) is None:
            missing.append(tool)
    if missing:
        return False, (
            f"llama-server not built yet and missing build tools: {', '.join(missing)}. "
            f"Install them or pre-build at {_binary_path()}"
        )
    return True, "llama-server not built yet — will clone + build on first run (~few min)"
