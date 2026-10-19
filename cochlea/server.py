import json
import struct
import time
from io import BytesIO

import soundfile
from flask import Flask, Response, jsonify, request, send_from_directory
from mo_files import File
from mo_logs import logger

from cochlea.harmonics import suppress
from cochlea.spectrogram import BINS, BINS_PER_OCTAVE, FMIN, magnitude, normalize

WEB = (File(__file__).parent.parent / "web").os_path

app = Flask(__name__, static_folder=None)

SUPPRESS = {"0": lambda linear, samples, sample_rate: linear, "1": suppress}  # ?suppress= value -> magnitude filter


def decode(data):
    """Return (mono float32 samples, sample_rate, channels)."""
    samples, sample_rate = soundfile.read(BytesIO(data), dtype="float32", always_2d=True)
    return samples.mean(axis=1), sample_rate, samples.shape[1]


def pack(samples, sample_rate, shape):
    """Analyze mono samples into the /analyze wire format; `shape` filters linear magnitude."""
    hop, linear = magnitude(samples, sample_rate)
    frames = normalize(shape(linear, samples=samples, sample_rate=sample_rate))
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
    shape = SUPPRESS.get(request.args.get("suppress", "0"))
    if shape is None:
        return jsonify(error="suppress must be 0 or 1"), 400
    start = time.monotonic()
    data = upload.read()
    logger.info("analyze: received {file} ({bytes} bytes)", file=upload.filename, bytes=len(data))
    try:
        mono, sample_rate, channels = decode(data)
    except Exception as cause:
        logger.warning("analyze: cannot decode {file}", file=upload.filename, cause=cause)
        return jsonify(error=f"cannot decode audio: {cause}"), 400
    logger.info(
        "analyze: decoded {seconds|round(places=1)}s at {rate} Hz, {channels} channels",
        seconds=len(mono) / sample_rate,
        rate=sample_rate,
        channels=channels,
    )
    body = pack(mono, sample_rate, shape)
    logger.info(
        "analyze: done in {seconds|round(places=1)}s, sending {bytes} bytes",
        seconds=time.monotonic() - start,
        bytes=len(body),
    )
    return Response(body, mimetype="application/octet-stream")
