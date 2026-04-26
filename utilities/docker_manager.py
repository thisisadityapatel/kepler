"""Docker image management for vLLM, llama.cpp, and TensorRT-LLM backends."""

import subprocess
import threading
import time
import requests
from pathlib import Path
from typing import Optional

from common import BACKEND_REGISTRY


def _stream_process_output(process: subprocess.Popen, prefix: str = "") -> None:
    """Read process stdout line-by-line and print with optional prefix."""
    try:
        for line in iter(process.stdout.readline, ""):
            print(f"{prefix}{line}", end="", flush=True)
    except Exception as exc:
        print(f"[debug] output reader error: {exc}", flush=True)


def _run_docker(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout: int | None = None,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a docker CLI command with better failure messages."""
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            "Docker command timed out. Is Docker Desktop running and responsive?\n"
            f"Command: {' '.join(cmd)}"
        ) from e
    except FileNotFoundError as e:
        raise RuntimeError(
            "Docker CLI not found. Install Docker Desktop (or ensure `docker` is on PATH)."
        ) from e


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


def build_llama_docker_conatiner_cmd(
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


class DockerContainer:
    """Manages Docker container lifecycle for LLM serving."""

    def __init__(self, name: str, image: str, port: int):
        self.name = name
        self.image = image
        self.port = port
        self.container_id: Optional[str] = None
        self.process: Optional[subprocess.Popen] = None

    def build_image(self, dockerfile_path: Path, version: str) -> bool:
        """Build Docker image if it doesn't exist."""
        check_cmd = ["docker", "images", "-q", f"{self.image}:{version}"]
        print(f"[debug] checking for existing image: {' '.join(check_cmd)}", flush=True)
        result = _run_docker(check_cmd, timeout=10)
        print(
            f"[debug] image check returncode={result.returncode} stdout={result.stdout.strip()!r}",
            flush=True,
        )

        if result.stdout.strip():
            print(
                f"[debug] image {self.image}:{version} already exists, skipping build",
                flush=True,
            )
            return True
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(
                "Docker is not accessible (failed to query local images). "
                "Make sure Docker Desktop is running and you have permission to access the Docker socket.\n"
                f"Command: {' '.join(check_cmd)}"
                + (f"\nError: {stderr}" if stderr else "")
            )

        build_cmd = [
            "docker",
            "build",
            "-f",
            str(dockerfile_path),
            "--build-arg",
            f"VERSION={version}",
            "-t",
            f"{self.image}:{version}",
            ".",
        ]

        print("[debug] building image (this may take several minutes)...", flush=True)
        print(f"[debug] build command: {' '.join(build_cmd)}", flush=True)
        print(f"[debug] build cwd: {dockerfile_path.parent.parent}", flush=True)

        try:
            build_process = subprocess.Popen(
                build_cmd,
                cwd=str(dockerfile_path.parent.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            print("[debug] docker build process started, streaming output:", flush=True)
            _stream_process_output(build_process, prefix="  [build] ")
            try:
                build_process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                build_process.kill()
                build_process.wait()
                print("[debug] docker build timed out after 1800s", flush=True)
                return False
            print(
                f"[debug] docker build finished with returncode={build_process.returncode}",
                flush=True,
            )
            if build_process.returncode == 0:
                return True
            raise RuntimeError(
                f"Failed to build Docker image (returncode={build_process.returncode}).\n"
                f"Command: {' '.join(build_cmd)}"
            )
        except Exception as exc:
            print(f"[debug] exception during build_image: {exc}", flush=True)
            raise

    def start_container(self, docker_cmd: list[str]) -> bool:
        """Start the Docker container."""
        print("[debug] starting container with command:", flush=True)
        print(f"  {' '.join(docker_cmd)}", flush=True)
        try:
            self.process = subprocess.Popen(
                docker_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            print(
                f"[debug] container process started (pid={self.process.pid})",
                flush=True,
            )
            reader = threading.Thread(
                target=_stream_process_output,
                args=(self.process,),
                kwargs={"prefix": "  [container] "},
                daemon=True,
            )
            reader.start()
            print("[debug] waiting 2s to check if process stays alive...", flush=True)
            time.sleep(2)
            poll = self.process.poll()
            print(
                f"[debug] process poll after 2s: {poll} (None = still running)",
                flush=True,
            )
            return poll is None
        except Exception as exc:
            print(f"[debug] exception in start_container: {exc}", flush=True)
            return False

    def wait_for_ready(self, timeout: int = 60) -> bool:
        """Wait for the model server to be ready."""
        completion_url = f"http://localhost:{self.port}/completion"
        print(
            f"[debug] waiting for server at {completion_url} (timeout={timeout}s)",
            flush=True,
        )
        start_time = time.time()
        last_status_print = start_time
        attempt = 0
        while time.time() - start_time < timeout:
            attempt += 1
            elapsed = time.time() - start_time
            if self.process and self.process.poll() is not None:
                print(
                    f"[debug] container process exited early (returncode={self.process.poll()}) after {elapsed:.1f}s",
                    flush=True,
                )
                return False
            try:
                test_payload = {"prompt": "test", "n_predict": 1}
                response = requests.post(completion_url, json=test_payload, timeout=5)
                print(
                    f"[debug] health check attempt {attempt}: status={response.status_code} elapsed={elapsed:.1f}s",
                    flush=True,
                )
                if response.status_code in [200, 400]:
                    print(f"[debug] server ready after {elapsed:.1f}s", flush=True)
                    return True
            except requests.exceptions.ConnectionError as exc:
                if time.time() - last_status_print >= 10:
                    print(
                        f"[debug] health check attempt {attempt}: connection refused at {elapsed:.1f}s — server still starting ({exc})",
                        flush=True,
                    )
                    last_status_print = time.time()
            except requests.exceptions.RequestException as exc:
                print(
                    f"[debug] health check attempt {attempt}: request error at {elapsed:.1f}s — {exc}",
                    flush=True,
                )
            time.sleep(2)
        print(f"[debug] server did not become ready within {timeout}s", flush=True)
        return False

    def stop_container(self) -> bool:
        """Stop the running container."""
        if self.process:
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                self.process = None
                return True
            except Exception:
                return False
        return True

    def is_running(self) -> bool:
        """Check if container is still running."""
        return self.process is not None and self.process.poll() is None


def create_llama_container(
    model_path: Path, port: int = 8080, ctx_size: int = 4096, version: str = "b7531"
) -> tuple[DockerContainer, list[str]]:
    """Create and configure a llama.cpp container."""
    image_name = "kepler-llama"
    container = DockerContainer(f"kepler-llama-{port}", image_name, port)

    # Build image if needed
    dockerfile_path = BACKEND_REGISTRY["llama"]["dockerfile"]
    if not container.build_image(dockerfile_path, version):
        raise RuntimeError("Failed to build Docker image")

    # Create Docker command
    docker_cmd = build_llama_docker_conatiner_cmd(
        image_name=f"{image_name}:{version}",
        gguf_path=str(model_path),
        port=port,
        ctx=ctx_size,
        n_gpu_layers=999,  # Use all available layers
    )

    return container, docker_cmd
