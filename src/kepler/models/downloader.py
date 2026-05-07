from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

from kepler.engines.base import ModelFormat

GGUF_QUANT_PRIORITY = ("Q5_K_M", "Q4_K_M", "Q8_0", "Q6_K", "Q4_0")


def repo_exists(repo_id: str) -> bool:
    try:
        HfApi().repo_info(repo_id)
        return True
    except (RepositoryNotFoundError, HfHubHTTPError):
        return False
    except Exception:
        return False


def list_repo_files(repo_id: str) -> list[str]:
    try:
        return list(HfApi().list_repo_files(repo_id))
    except Exception:
        return []


def pick_gguf_file(files: list[str]) -> str | None:
    """Pick the best GGUF artifact: prefer Q5_K_M, then Q4_K_M, then highest-precision.
    Match is case-insensitive — repos vary on capitalization (Q5_K_M vs q5_k_m)."""
    ggufs = [f for f in files if f.lower().endswith(".gguf")]
    if not ggufs:
        return None
    for tag in GGUF_QUANT_PRIORITY:
        tag_lower = tag.lower()
        matches = [f for f in ggufs if tag_lower in f.lower()]
        if matches:
            shards = [f for f in matches if "-of-" in f]
            if shards:
                return _first_shard(shards)
            return sorted(matches)[0]
    shards = [f for f in ggufs if "-of-" in f]
    if shards:
        return _first_shard(shards)
    return sorted(ggufs)[0]


def _first_shard(shards: list[str]) -> str:
    for s in sorted(shards):
        if "-00001-of-" in s:
            return s
    return sorted(shards)[0]


def download_gguf(
    repo_id: str,
    filename: str,
    models_dir: Path,
    progress_cb=None,
) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(models_dir),
        token=os.environ.get("HF_TOKEN"),
    )
    return Path(path)


def download_snapshot(
    repo_id: str,
    models_dir: Path,
    allow_patterns: list[str] | None = None,
) -> Path:
    target = models_dir / repo_id.replace("/", "__")
    target.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        allow_patterns=allow_patterns,
        token=os.environ.get("HF_TOKEN"),
    )
    return Path(path)


def detect_local_format(path: Path) -> ModelFormat | None:
    if path.is_file() and path.suffix.lower() == ".gguf":
        return ModelFormat.GGUF
    if path.is_dir():
        if any(p.suffix == ".gguf" for p in path.iterdir()):
            return ModelFormat.GGUF
        if (path / "config.json").exists():
            return ModelFormat.MLX
    return None
