from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

CHILD_JOIN_TIMEOUT_S = 10
RELEASE_VERIFY_TIMEOUT_S = 30


@dataclass
class ChildOutcome:
    ok: bool
    payload: object | None
    error: str | None


def run_in_subprocess(target, args: tuple, timeout_s: float = 3600.0) -> ChildOutcome:
    """Spawn a fresh process (spawn ctx, no inherited fds), run `target(*args, conn)`,
    collect the outcome over a Pipe. The child must call conn.send({"ok": ..., "payload": ...,
    "error": ...}) exactly once before exiting. We rely on process exit for GPU cleanup."""
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child_entrypoint, args=(target, args, child_conn), daemon=False)
    proc.start()
    child_conn.close()
    try:
        if parent_conn.poll(timeout_s):
            msg = parent_conn.recv()
            outcome = ChildOutcome(
                ok=bool(msg.get("ok", False)),
                payload=msg.get("payload"),
                error=msg.get("error"),
            )
        else:
            outcome = ChildOutcome(ok=False, payload=None, error=f"child timed out after {timeout_s}s")
    except EOFError:
        outcome = ChildOutcome(ok=False, payload=None, error="child exited without sending result")
    finally:
        try:
            parent_conn.close()
        except Exception:
            pass
        _reap(proc)
    return outcome


def _child_entrypoint(target, args, conn):
    try:
        result = target(*args)
        conn.send({"ok": True, "payload": result})
    except BaseException as exc:
        import traceback

        conn.send(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            }
        )
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _reap(proc) -> None:
    if proc is None:
        return
    proc.join(timeout=CHILD_JOIN_TIMEOUT_S)
    if proc.is_alive():
        try:
            proc.terminate()
            proc.join(timeout=5)
        except Exception:
            pass
    if proc.is_alive():
        try:
            proc.kill()
            proc.join(timeout=5)
        except Exception:
            pass


def verify_released(child_pid: int | None, port: int | None = None) -> bool:
    """Best-effort check: child PID gone, no descendant processes still alive,
    optional port released. Used to gate the next engine."""
    deadline = time.monotonic() + RELEASE_VERIFY_TIMEOUT_S
    while time.monotonic() < deadline:
        alive_descendants: list[int] = []
        if child_pid is not None:
            try:
                p = psutil.Process(child_pid)
                if p.is_running():
                    alive_descendants.append(child_pid)
                else:
                    alive_descendants.extend(c.pid for c in p.children(recursive=True) if c.is_running())
            except psutil.NoSuchProcess:
                pass
        port_clear = True
        if port is not None:
            port_clear = not _port_in_use(port)
        if not alive_descendants and port_clear:
            return True
        time.sleep(0.5)
    return False


def _port_in_use(port: int) -> bool:
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        try:
            s.close()
        except Exception:
            pass


def find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def cooldown(seconds: int, on_tick=None) -> None:
    """Sleep `seconds` total. If on_tick is provided, call on_tick(remaining) every second."""
    if seconds <= 0:
        return
    end = time.monotonic() + seconds
    while True:
        remaining = end - time.monotonic()
        if remaining <= 0:
            break
        if on_tick is not None:
            on_tick(int(round(remaining)))
        time.sleep(min(1.0, remaining))


def write_partial_marker(perf_dir: Path, comparison_id: str) -> Path:
    """Drop a small breadcrumb so a partial comparison run can be identified later."""
    perf_dir.mkdir(parents=True, exist_ok=True)
    p = perf_dir / f"comparison_{comparison_id}.partial"
    p.write_text("interrupted\n")
    return p
