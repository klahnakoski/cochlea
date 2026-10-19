"""Chart the Notes view on test windows of a song, and time it on the whole song.

Run from the repo root: python -m scripts.rectangles [song.wav [chart folder]]
Writes charts/rect-<window>.png panels: image | beat mask | fitted rectangles | after overtone folding.
"""

import struct
import sys
import time
import zlib

import numpy
import soundfile
from mo_files import File
from mo_logs import logger

from cochlea.notes import ROWS, ROWS_PER_SEMITONE, image, loudness, notes, painted, per_key, stages
from cochlea.spectrogram import BINS, FRAME_RATE

SONG = "~/Music/Chris Sheppard - Pirate Radio Sessions Volume 4/04 Can't Love You (B.G. The Prince Of Rap).wav"
WINDOWS = {"0m00-0m06": (0, 6), "1m04-1m08": (64, 68), "2m53-3m00": (173, 180)}  # name -> (first, last) seconds
MARGIN_SECONDS = 0.5  # read past each edge so the fit has context
WIDE = 3  # chart pixels per frame
TALL = 2  # chart pixels per row


def drawn(rectangles, frames, level):
    """(ROWS, frames) rectangles at their `level` across their key's rows, thinner when quieter."""
    filled = numpy.repeat(painted(rectangles, frames=frames, level=level), ROWS_PER_SEMITONE, axis=0)
    inner = numpy.arange(ROWS) % ROWS_PER_SEMITONE - (ROWS_PER_SEMITONE - 1) / 2  # row offset from the key's centre
    return numpy.where(numpy.abs(inner)[:, None] <= filled * ROWS_PER_SEMITONE / 2, filled, 0)


def picture(grey):
    """Rows flipped so C1 is at the bottom, pixels enlarged, a dotted line under every C."""
    pixels = (numpy.repeat(numpy.repeat(grey[::-1], TALL, axis=0), WIDE, axis=1) * 255).astype(numpy.uint8)
    for octave in range(8):
        pixels[TALL * (ROWS - octave * 12 * ROWS_PER_SEMITONE) - 1, ::3] = 110
    return pixels


def write_png(path, grey):
    height, width = grey.shape
    raw = b"".join(b"\x00" + numpy.repeat(grey[y, :, None], 3, axis=1).tobytes() for y in range(height))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    File(path).write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    )


def chart(song, output, name, first, last):
    rate = soundfile.info(song).samplerate
    begin = max(0, first - MARGIN_SECONDS)
    samples, _ = soundfile.read(
        song, dtype="float32", always_2d=True, start=round(begin * rate), stop=round((last + MARGIN_SECONDS) * rate)
    )
    frames = -(-len(samples) // round(rate / FRAME_RATE))
    step = stages(samples.mean(axis=1), sample_rate=rate, frames=frames)
    rectangles, keep = step["rectangles"], step["keep"]
    kept = {key: column[keep] for key, column in rectangles.items()}
    level = loudness(rectangles, grey=step["key_grey"], known=step["key_known"])
    lit = painted(kept, frames=frames, level=level[keep]).T
    crop = slice(round((first - begin) * FRAME_RATE), round((last - begin) * FRAME_RATE))
    inside = (rectangles["end"] >= crop.start) & (rectangles["start"] < crop.stop)
    contrast = per_key(step["contrast"]).max(axis=1)
    explained = (rectangles["total"] ** 2 / rectangles["weight"]).sum() / (step["key_known"] * contrast**2).sum()
    panels = [
        step["grey"],
        numpy.where(step["known"], 0, step["grey"]),
        drawn(rectangles, frames=frames, level=level),
        drawn(kept, frames=frames, level=level[keep]),
    ]
    gap = numpy.full((TALL * ROWS, 6), 60, numpy.uint8)
    joined = [part for panel in panels for part in (picture(panel[:, crop]), gap)][:-1]
    path = output / f"rect-{name}.png"
    write_png(path.os_path, numpy.concatenate(joined, axis=1))
    seconds = last - first
    logger.info(
        "{name}: {fitted|round(digits=3)} rectangles/s fitted, {kept|round(digits=3)}/s after folding,"
        " {keys|round(digits=3)} keys lit per frame, {explained|round(digits=3)} of known energy explained"
        " (whole read), {unknown|round(digits=3)} of pixels unknown -> {path}",
        name=name,
        fitted=inside.sum() / seconds,
        kept=(inside & keep).sum() / seconds,
        keys=(lit[crop] > 0).sum(axis=1).mean(),
        explained=explained,
        unknown=1 - step["known"][:, crop].mean(),
        path=path.os_path,
    )


def timing(song):
    samples, rate = soundfile.read(song, dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    frames = -(-len(samples) // round(rate / FRAME_RATE))
    start = time.monotonic()
    image(samples, sample_rate=rate, frames=frames)
    imaged = time.monotonic()
    lit, fields = notes(numpy.zeros((frames, BINS)), samples=samples, sample_rate=rate)
    logger.info(
        "whole song {seconds|round(digits=3)}s: notes view {total|round(digits=3)}s, of which the image"
        " {image|round(digits=3)}s; {rectangles} rectangles ({per|round(digits=3)}/s), {keys|round(digits=3)} keys"
        " lit per frame",
        seconds=len(samples) / rate,
        total=time.monotonic() - imaged,
        image=imaged - start,
        rectangles=fields["rectangles"],
        per=fields["rectangles"] * rate / len(samples),
        keys=(lit > 0).sum(axis=1).mean(),
    )


def main():
    logger.start()
    song = File(sys.argv[1] if len(sys.argv) > 1 else SONG).os_path
    output = File(sys.argv[2]) if len(sys.argv) > 2 else File(__file__).parent.parent / "charts"
    for name, (first, last) in WINDOWS.items():
        chart(song, output=output, name=name, first=first, last=last)
    timing(song)


if __name__ == "__main__":
    main()
