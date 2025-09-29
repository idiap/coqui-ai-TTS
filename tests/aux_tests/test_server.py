import os
import signal
import socket
import subprocess
import time
import wave

import pytest
import requests

PORT = 5003


def wait_for_server(host, port, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except (OSError, ConnectionRefusedError):
            time.sleep(1)
    raise TimeoutError(f"Server at {host}:{port} did not start within {timeout} seconds.")


@pytest.fixture(scope="module", autouse=True)
def start_flask_server():
    server_process = subprocess.Popen(
        ["python", "-m", "TTS.server.server", "--port", str(PORT)],
    )
    wait_for_server("localhost", PORT)
    yield
    os.kill(server_process.pid, signal.SIGTERM)
    server_process.wait()


def test_flask_server(tmp_path):
    url = f"http://localhost:{PORT}/api/tts?text=synthesis%20schmynthesis"
    response = requests.get(url)
    assert response.status_code == 200, f"Request failed with status code {response.status_code}"

    wav_path = tmp_path / "output.wav"
    with wav_path.open("wb") as f:
        f.write(response.content)

    with wave.open(str(wav_path), "rb") as wav_file:
        num_frames = wav_file.getnframes()
        assert num_frames > 0, "WAV file contains no frames."
