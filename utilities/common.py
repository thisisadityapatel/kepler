from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "performances"
PERF_ROOT = ROOT / "perf"
EVAL_RESULTS_ROOT = ROOT / "evaluations"
CONFIG_PATH = ROOT / "config" / "models.yaml"
BACKEND_REGISTRY = {
    "llama": {
        "display_name": "llama.cpp",
        "formats": ["gguf"],
        "default_port": 8080,
        "image_prefix": "kepler-llama",
        "prebuilt_image": None,
        "dockerfile": ROOT / "docker" / "Dockerfile.llama",
        "docker_base": ["--rm"],
    }
}
