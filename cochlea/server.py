import json
import struct
import time
from io import BytesIO

import soundfile
from flask import Flask, Response, jsonify, request, send_from_directory
from mo_files import File
from mo_logs import logger

from cochlea.drums import kicks
from cochlea.harmonics import suppress
from cochlea.spectrogram import BINS, BINS_PER_OCTAVE, FMIN, emphasize, magnitude, normalize
from cochlea.voices import top

WEB = (File(__file__).parent.parent / "web").os_path

app = Flask(__name__, static_folder=None)

SUPPRESS = {"0": lambda linear, samples, sample_rate: linear, "1": suppress}  # ?suppress= value -> magnitude filter


def top_notes(levels, samples, sample_rate):
    lit, voices = top(levels, samples=samples, sample_rate=sample_rate)
    return lit, {"voices": voices}


VIEWS = {"energy": lambda levels, samples, sample_rate: (levels, {}), "top": top_notes}  # ?view= -> (levels, header)


def decode(data):
    """Return (mono float32 samples, sample_rate, channels)."""
    samples, sample_rate = soundfile.read(BytesIO(data), dtype="float32", always_2d=True)
    return samples.mean(axis=1), sample_rate, samples.shape[1]


def pack(samples, sample_rate, shape, view):
    """Analyze mono samples into the /analyze wire format; `shape` filters linear magnitude, `view` the levels."""
    hop, linear = magnitude(samples, sample_rate)
    levels = normalize(shape(emphasize(linear), samples=samples, sample_rate=sample_rate))
    frames, fields = view(levels, samples=samples, sample_rate=sample_rate)
    hit_frames, hit_levels = kicks(samples, sample_rate)
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
            "drums": {"frame": hit_frames.tolist(), "level": hit_levels.round(3).tolist()},
            **fields,
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
    view = VIEWS.get(request.args.get("view", "energy"))
    if view is None:
        return jsonify(error="view must be energy or top"), 400
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
    body = pack(mono, sample_rate=sample_rate, shape=shape, view=view)
    logger.info(
        "analyze: done in {seconds|round(places=1)}s, sending {bytes} bytes",
        seconds=time.monotonic() - start,
        bytes=len(body),
    )
    return Response(body, mimetype="application/octet-stream")
