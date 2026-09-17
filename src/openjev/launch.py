"""Small stdlib-only helpers usable by Modal's base Python environment."""

import os
import signal
import subprocess
import time
import urllib.error
import urllib.request


def start_background(python: str, sglang_python: str, timeout: float = 1200):
    process = subprocess.Popen(
        [python, "-m", "openjev", "serve", "--host", "0.0.0.0", "--sglang-python", sglang_python],
        start_new_session=True,
    )
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"OpenJev exited with status {process.returncode}")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as response:
                    if response.status == 200:
                        return process
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(1)
        raise TimeoutError("OpenJev did not start before the startup deadline")
    except BaseException:
        stop_background(process)
        raise


def stop_background(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=25)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
