import json
import struct
from io import BytesIO

import numpy
import pytest
import soundfile

from cochlea.server import app, decode
from cochlea.spectrogram import FMIN

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
    assert response.mimetype == "application/octet-stream"
    body = response.data
    (length,) = struct.unpack("<I", body[:4])
    header = json.loads(body[4 : 4 + length])
    assert header == {
        "sample_rate": RATE,
        "hop": 735,
        "frame_rate": 60.0,
        "fmin": FMIN,
        "bins_per_octave": 12,
        "bins": 96,
        "frames": 60,
        "duration": 1.0,
    }
    values = numpy.frombuffer(body[4 + length :], dtype="<f4")
    assert values.size == 60 * 96
    frames = values.reshape(60, 96)
    # last frames look ahead past the end of the file
    assert frames[:-5].argmax(axis=1).tolist() == [45] * 55
    assert 0 <= values.min() and values.max() == 1


def test_analyze_suppress_keeps_wire_format(client):
    plain = client.post("/analyze", data={"audio": (BytesIO(sine_wav(1)), "a.wav")}).data
    suppressed = client.post("/analyze?suppress=1", data={"audio": (BytesIO(sine_wav(1)), "a.wav")}).data
    (length,) = struct.unpack("<I", plain[:4])
    assert suppressed[: 4 + length] == plain[: 4 + length]
    frames = numpy.frombuffer(suppressed[4 + length :], dtype="<f4").reshape(60, 96)
    assert frames[:-5].argmax(axis=1).tolist() == [45] * 55


def test_analyze_rejects_unknown_suppress(client):
    response = client.post("/analyze?suppress=2", data={"audio": (BytesIO(sine_wav(1)), "a.wav")})
    assert response.status_code == 400


def test_analyze_rejects_garbage(client):
    response = client.post("/analyze", data={"audio": (BytesIO(b"not audio"), "a.wav")})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_decode_mono():
    mono, sample_rate, channels = decode(sine_wav(1))
    assert (mono.shape, sample_rate, channels) == ((RATE,), RATE, 1)


def test_index_served(client):
    assert b"Cochlea" in client.get("/").data
