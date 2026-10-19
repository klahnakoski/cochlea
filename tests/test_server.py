import json
import struct
from io import BytesIO

import numpy
import pytest
import soundfile

from mo_files import File

from cochlea import server
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
        "drums": {"frame": [], "level": []},
    }
    values = numpy.frombuffer(body[4 + length :], dtype="<f4")
    assert values.size == 60 * 96
    frames = values.reshape(60, 96)
    # first frames are the attack
    assert frames[3:].argmax(axis=1).tolist() == [45] * 57
    assert 0 <= values.min() and values.max() == 1


def test_analyze_suppress_keeps_wire_format(client):
    plain = client.post("/analyze", data={"audio": (BytesIO(sine_wav(1)), "a.wav")}).data
    suppressed = client.post("/analyze?suppress=1", data={"audio": (BytesIO(sine_wav(1)), "a.wav")}).data
    (length,) = struct.unpack("<I", plain[:4])
    assert suppressed[: 4 + length] == plain[: 4 + length]
    frames = numpy.frombuffer(suppressed[4 + length :], dtype="<f4").reshape(60, 96)
    assert frames[3:].argmax(axis=1).tolist() == [45] * 57


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


def test_analyze_reports_kick_frames_and_levels(client):
    t = numpy.arange(round(0.3 * RATE)) / RATE
    kick = numpy.sin(2 * numpy.pi * 50 * t) * numpy.exp(-t / 0.1)
    wave = numpy.zeros(4 * RATE)
    for second in range(4):
        wave[second * RATE + RATE // 2 :][: len(t)] = kick
    buffer = BytesIO()
    soundfile.write(buffer, wave.astype("float32"), RATE, format="WAV")
    response = client.post("/analyze", data={"audio": (BytesIO(buffer.getvalue()), "k.wav")})
    (length,) = struct.unpack("<I", response.data[:4])
    drums = json.loads(response.data[4 : 4 + length])["drums"]
    assert len(drums["frame"]) == 4 and len(drums["level"]) == 4
    assert [abs(f - (30 + 60 * i)) <= 1 for i, f in enumerate(drums["frame"])] == [True] * 4
    assert max(drums["level"]) == 1


def test_analyze_top_reports_voices(client):
    response = client.post("/analyze?view=top", data={"audio": (BytesIO(sine_wav(1)), "a.wav")})
    (length,) = struct.unpack("<I", response.data[:4])
    header = json.loads(response.data[4 : 4 + length])
    assert 2 <= header["voices"] <= 8
    frames = numpy.frombuffer(response.data[4 + length :], dtype="<f4").reshape(header["frames"], 96)
    assert ((frames > 0).sum(axis=1) <= header["voices"]).all()
    assert (frames[10:].argmax(axis=1) == 45).all()


def test_analyze_rejects_unknown_view(client):
    response = client.post("/analyze?view=hits", data={"audio": (BytesIO(sine_wav(1)), "a.wav")})
    assert response.status_code == 400


def test_tracks_lists_and_serves_the_music_folder(client, tmp_path, monkeypatch):
    (tmp_path / "b.wav").write_bytes(sine_wav(1))
    (tmp_path / "a.flac").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("not audio")
    monkeypatch.setattr(server, "MUSIC", File(str(tmp_path)))
    assert client.get("/tracks").json == ["a.flac", "b.wav"]
    assert client.get("/tracks/b.wav").data == sine_wav(1)


def test_analyze_notes_reports_rectangles(client):
    response = client.post("/analyze?view=notes", data={"audio": (BytesIO(sine_wav(1)), "a.wav")})
    (length,) = struct.unpack("<I", response.data[:4])
    header = json.loads(response.data[4 : 4 + length])
    assert header["rectangles"] >= 1
    frames = numpy.frombuffer(response.data[4 + length :], dtype="<f4").reshape(header["frames"], 96)
    assert numpy.flatnonzero((frames[15:45] > 0).any(axis=0)).tolist() == [45]
