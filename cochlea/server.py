from io import BytesIO
from pathlib import Path

import soundfile
from flask import Flask, jsonify, request, send_from_directory

WEB = Path(__file__).resolve().parent.parent / "web"

app = Flask(__name__, static_folder=None)


def decode(data):
    """Return (mono float32 samples, sample_rate, channels)."""
    samples, sample_rate = soundfile.read(BytesIO(data), dtype="float32", always_2d=True)
    return samples.mean(axis=1), sample_rate, samples.shape[1]


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
        mono, sample_rate, channels = decode(upload.read())
    except Exception as cause:
        return jsonify(error=f"cannot decode audio: {cause}"), 400
    return jsonify(
        sample_rate=sample_rate,
        channels=channels,
        duration=len(mono) / sample_rate,
        samples=len(mono),
    )
