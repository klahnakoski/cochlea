import json
import struct
from io import BytesIO

import soundfile
from flask import Flask, Response, jsonify, request, send_from_directory
from mo_files import File

from cochlea.spectrogram import BINS, BINS_PER_OCTAVE, FMIN, spectrogram

WEB = (File(__file__).parent.parent / "web").os_path

app = Flask(__name__, static_folder=None)


def decode(data):
    """Return (mono float32 samples, sample_rate, channels)."""
    samples, sample_rate = soundfile.read(
        BytesIO(data), dtype="float32", always_2d=True
    )
    return samples.mean(axis=1), sample_rate, samples.shape[1]


def pack(samples, sample_rate):
    """Analyze mono samples into the /analyze wire format."""
    hop, frames = spectrogram(samples, sample_rate)
    header = (
        json
        .dumps({
            "sample_rate": sample_rate,
            "hop": hop,
            "frame_rate": sample_rate / hop,
            "fmin": FMIN,
            "bins_per_octave": BINS_PER_OCTAVE,
            "bins": BINS,
            "frames": len(frames),
            "duration": len(samples) / sample_rate,
        })
        .encode("utf-8")
    )
    return struct.pack("<I", len(header)) + header + frames.astype("<f4").tobytes()


@app.get("/")
def index():
    return send_from_directory(WEB, "index.html")


@app.get("/<path:path>")
def static_file(path):
    return send_from_directory(WEB, path)


@app.post("/analyze")
def analyze():
    upload = request.files.get("audio")
    if upload is None:
        return jsonify(error="missing form field 'audio'"), 400
    try:
        mono, sample_rate, _ = decode(upload.read())
    except Exception as cause:
        return jsonify(error=f"cannot decode audio: {cause}"), 400
    return Response(pack(mono, sample_rate), mimetype="application/octet-stream")
