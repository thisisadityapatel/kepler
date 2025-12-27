"""Docker image management for vLLM, llama.cpp, and TensorRT-LLM backends."""

from pathlib import Path

from common import BACKEND_REGISTRY


def _docker_run_base(
    engine: str,
    image_name: str,
    port: int,
    mounts: list[tuple[str, str, str]],
) -> list[str]:
    """Build common Docker run prefix.

    Args:
        engine: Backend name (uses BACKEND_REGISTRY for base flags)
        image_name: Docker image to use
        port: Port to expose
        mounts: List of (host_path, container_path, mode) tuples

    Returns:
        Docker run command prefix (up to and including image name)
    """
    cfg = BACKEND_REGISTRY[engine]
    cmd = ["docker", "run", "--rm"]
    cmd.extend(cfg["docker_base"])
    cmd.extend(["-p", f"{port}:{port}"])
    for host_path, container_path, mode in mounts:
        cmd.extend(["-v", f"{host_path}:{container_path}:{mode}"])
    cmd.append(image_name)
    return cmd


def build_llama_docker_cmd(
    image_name: str,
    gguf_path: str,
    port: int,
    n_gpu_layers: int | None = None,
    ctx: int | None = None,
    parallel: int | None = None,
    mmproj_path: str | None = None,
    repeat_penalty: float | None = None,
    repeat_last_n: int | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Build Docker run command for llama-server."""
    gguf_path_resolved = str(Path(gguf_path).expanduser().resolve())
    model_dir = str(Path(gguf_path_resolved).parent)

    mounts = [(model_dir, model_dir, "ro")]
    if mmproj_path:
        mmproj_resolved = str(Path(mmproj_path).expanduser().resolve())
        mmproj_dir = str(Path(mmproj_resolved).parent)
        if mmproj_dir != model_dir:
            mounts.append((mmproj_dir, mmproj_dir, "ro"))

    cmd = _docker_run_base("llama", image_name, port, mounts)
    cmd += ["-m", gguf_path_resolved, "--host", "0.0.0.0", "--port", str(port), "-v"]

    if n_gpu_layers is not None:
        cmd += ["-ngl", str(n_gpu_layers)]
    if ctx is not None:
        cmd += ["-c", str(ctx)]
    if parallel is not None and parallel > 1:
        cmd += ["-np", str(parallel)]
    if mmproj_path:
        cmd += ["--mmproj", str(Path(mmproj_path).expanduser().resolve())]
    if repeat_penalty is not None:
        cmd += ["--repeat-penalty", str(repeat_penalty)]
    if repeat_last_n is not None:
        cmd += ["--repeat-last-n", str(repeat_last_n)]
    if extra_args:
        cmd += extra_args
    return cmd
