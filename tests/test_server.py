from io import BytesIO

import numpy
import pytest
import soundfile

from cochlea.server import app, decode

RATE = 44100


def sine_wav(channels):
    t = numpy.arange(RATE) / RATE
    wave = numpy.sin(2 * numpy.pi * 440 * t).astype("float32")
    data = wave if channels == 1 else numpy.stack([wave] * channels, axis=1)
    buffer = BytesIO()
    soundfile.write(buffer, data, RATE, format="WAV")
    return buffer.getvalue()


@pytest.fixture
def client():
    return app.test_client()


def test_analyze_stereo(client):
    response = client.post("/analyze", data={"audio": (BytesIO(sine_wav(2)), "a.wav")})
    assert response.status_code == 200
    assert response.get_json() == {
        "sample_rate": RATE,
        "channels": 2,
        "duration": 1.0,
        "samples": RATE,
    }


def test_analyze_rejects_garbage(client):
    response = client.post("/analyze", data={"audio": (BytesIO(b"not audio"), "a.wav")})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_decode_mono():
    mono, sample_rate, channels = decode(sine_wav(1))
    assert (mono.shape, sample_rate, channels) == ((RATE,), RATE, 1)


def test_index_served(client):
    assert b"Cochlea" in client.get("/").data
