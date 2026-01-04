"""Docker image management for vLLM, llama.cpp, and TensorRT-LLM backends."""

import subprocess
import time
import requests
from pathlib import Path
from typing import Optional, Tuple

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
        # Check if image exists
        check_cmd = ["docker", "images", "-q", f"{self.image}:{version}"]
        result = subprocess.run(check_cmd, capture_output=True, text=True)
        
        if result.stdout.strip():
            print(f"-- Docker image {self.image}:{version} already exists")
            return True
        
        print(f"-- Building Docker image {self.image}:{version}...")
        build_cmd = [
            "docker", "build", 
            "-f", str(dockerfile_path),
            "--build-arg", f"VERSION={version}",
            "-t", f"{self.image}:{version}",
            "."
        ]
        
        print(f"-- Running command: {' '.join(build_cmd)}")
        
        try:
            process = subprocess.Popen(
                build_cmd, 
                cwd=dockerfile_path.parent.parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            # Wait for completion
            return_code = process.wait(timeout=600)
            
            if return_code == 0:
                print(f"-- Successfully built {self.image}:{version}")
                return True
            else:
                print(f"-- Failed to build image with return code: {return_code}")
                return False
                
        except subprocess.TimeoutExpired:
            print("-- Docker build timed out after 10 minutes")
            if 'process' in locals():
                process.kill()
            return False
        except Exception as e:
            print(f"-- Error building Docker image: {e}")
            return False
    
    def start_container(self, docker_cmd: list[str]) -> bool:
        """Start the Docker container."""
        print(f"-- Starting container with command: {' '.join(docker_cmd)}")
        
        try:
            self.process = subprocess.Popen(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Give it a moment to start
            time.sleep(2)
            
            # Check if process is still running
            if self.process.poll() is None:
                print(f"-- Container started successfully on port {self.port}")
                return True
            else:
                stdout, stderr = self.process.communicate()
                print(f"-- Container failed to start:")
                if stdout:
                    print(f"Output: {stdout}")
                if stderr:
                    print(f"Errors: {stderr}")
                return False
                
        except Exception as e:
            print(f"-- Error starting container: {e}")
            return False
    
    def wait_for_ready(self, timeout: int = 60) -> bool:
        """Wait for the model server to be ready."""
        print(f"-- Waiting for model server to be ready on port {self.port}...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Try to connect to the health endpoint or completion endpoint
                health_url = f"http://localhost:{self.port}/health"
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    print(f"-- Model server is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            # If health endpoint doesn't exist, try completion endpoint
            try:
                completion_url = f"http://localhost:{self.port}/completion"
                test_payload = {"prompt": "test", "n_predict": 1}
                response = requests.post(completion_url, json=test_payload, timeout=5)
                if response.status_code in [200, 400]:  # 400 is OK, means server is responding
                    print(f"-- Model server is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            print(".", end="", flush=True)
            time.sleep(2)
        
        print(f"\n-- Model server did not become ready within {timeout} seconds")
        return False
    
    def stop_container(self) -> bool:
        """Stop the running container."""
        if self.process:
            print(f"-- Stopping container...")
            try:
                self.process.terminate()
                # Wait up to 10 seconds for graceful shutdown
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print("-- Force killing container...")
                    self.process.kill()
                    self.process.wait()
                
                print("-- Container stopped")
                self.process = None
                return True
            except Exception as e:
                print(f"-- Error stopping container: {e}")
                return False
        return True
    
    def is_running(self) -> bool:
        """Check if container is still running."""
        return self.process is not None and self.process.poll() is None


def create_llama_container(model_path: Path, port: int = 8080, ctx_size: int = 4096, 
                          version: str = "b7531") -> tuple[DockerContainer, list[str]]:
    """Create and configure a llama.cpp container."""
    image_name = f"kepler-llama"
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
        n_gpu_layers=999  # Use all available layers
    )
    
    return container, docker_cmd
