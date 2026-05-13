from __future__ import annotations

import platform
import subprocess


def get_system_info() -> dict[str, str]:
    """Replicates legacy/src/benchmark.cpp:68-83 — platform, architecture, processor."""
    info = {
        "platform": platform.system(),
        "architecture": platform.machine(),
        "processor": platform.processor() or "unknown",
    }
    if info["platform"] == "Darwin":
        info["processor"] = _macos_brand_string() or info["processor"]
    return info


def _macos_brand_string() -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None
