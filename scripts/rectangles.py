"""Chart the Notes view on test windows of a song, and time it on the whole song.

Run from the repo root: python -m scripts.rectangles [song.wav [chart folder]]
Writes charts/rect-<window>.png panels: image | known pixels (beats black) | fitted rectangles | after overtone folding;
held rectangles white, the voice pass amber.
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
WHITE = numpy.array([1.0, 1.0, 1.0])
AMBER = numpy.array([1.0, 0.7, 0.25])  # voice pass


def drawn(rectangles, frames, level):
    """(ROWS, frames) rectangles at their `level` across their key's rows, thinner when quieter."""
    filled = numpy.repeat(painted(rectangles, frames=frames, level=level), ROWS_PER_SEMITONE, axis=0)
    inner = numpy.arange(ROWS) % ROWS_PER_SEMITONE - (ROWS_PER_SEMITONE - 1) / 2  # row offset from the key's centre
    return numpy.where(numpy.abs(inner)[:, None] <= filled * ROWS_PER_SEMITONE / 2, filled, 0)


def picture(grey, voice):
    """RGB: `grey` white and `voice` amber, rows flipped so C1 is at the bottom, enlarged, a dotted line under every C."""
    shade = numpy.maximum(grey[..., None] * WHITE, voice[..., None] * AMBER)
    pixels = (numpy.repeat(numpy.repeat(shade[::-1], TALL, axis=0), WIDE, axis=1) * 255).astype(numpy.uint8)
    for octave in range(8):
        pixels[TALL * (ROWS - octave * 12 * ROWS_PER_SEMITONE) - 1, ::3] = 110
    return pixels


def write_png(path, rgb):
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(height))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    File(path).write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    )


def chained_strokes(rectangles, inside):
    """Strokes of two or more rectangles among those `inside`."""
    _, count = numpy.unique(rectangles["stroke"][inside], return_counts=True)
    return (count >= 2).sum()


def chart(song, output, name, first, last):
    rate = soundfile.info(song).samplerate
    begin = max(0, first - MARGIN_SECONDS)
    samples, _ = soundfile.read(
        song, dtype="float32", always_2d=True, start=round(begin * rate), stop=round((last + MARGIN_SECONDS) * rate)
    )
    frames = -(-len(samples) // round(rate / FRAME_RATE))
    step = stages(samples.mean(axis=1), sample_rate=rate, frames=frames)
    crop = slice(round((first - begin) * FRAME_RATE), round((last - begin) * FRAME_RATE))
    passes = {
        "held": (step["rectangles"], step["keep"], step["key_known"]),
        "voice": (step["voice"], step["voice_keep"], step["voice_known"]),
    }
    contrast = per_key(step["contrast"]).max(axis=1)
    known_energy = (step["key_known"] * contrast**2).sum()
    fitted, kept, stats = {}, {}, {}
    for which, (rectangles, keep, known) in passes.items():
        level = loudness(rectangles, grey=step["key_grey"], known=known)
        fitted[which] = drawn(rectangles, frames=frames, level=level)
        kept[which] = drawn({key: column[keep] for key, column in rectangles.items()}, frames=frames, level=level[keep])
        inside = (rectangles["end"] >= crop.start) & (rectangles["start"] < crop.stop)
        stats[which] = {
            "fitted": inside.sum(),
            "kept": (inside & keep).sum(),
            "strokes": chained_strokes(rectangles, inside & keep),
            "explained": (rectangles["total"] ** 2 / rectangles["weight"]).sum() / known_energy,
        }
    covered = numpy.repeat(step["covered"], ROWS_PER_SEMITONE, axis=0)
    for drawing in (fitted, kept):
        drawing["voice"] = numpy.where(covered, 0, drawing["voice"])
    zero = numpy.zeros_like(step["grey"])
    panels = [
        (step["grey"], zero),
        (numpy.where(step["known"], step["grey"], 0), zero),
        (fitted["held"], fitted["voice"]),
        (kept["held"], kept["voice"]),
    ]
    gap = numpy.full((TALL * ROWS, 6, 3), 60, numpy.uint8)
    joined = [part for grey, voice in panels for part in (picture(grey[:, crop], voice[:, crop]), gap)][:-1]
    path = output / f"rect-{name}.png"
    write_png(path.os_path, numpy.concatenate(joined, axis=1))
    seconds = last - first
    keys = {which: (per_key(drawing[:, crop]).max(axis=1) > 0).sum(axis=0).mean() for which, drawing in kept.items()}
    for which, stat in stats.items():
        logger.info(
            "{name} {which}: {fitted|round(digits=3)} rectangles/s fitted, {kept|round(digits=3)}/s after folding,"
            " {strokes|round(digits=3)} chained strokes/s, {keys|round(digits=3)} keys lit per frame,"
            " {explained|round(digits=3)} of known energy explained (whole read)",
            name=name,
            which=which,
            fitted=stat["fitted"] / seconds,
            kept=stat["kept"] / seconds,
            strokes=stat["strokes"] / seconds,
            keys=keys[which],
            explained=stat["explained"],
        )
    logger.info(
        "{name}: {unknown|round(digits=3)} of pixels unknown -> {path}",
        name=name,
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
    held, voice = lit[:frames], lit[frames:]
    logger.info(
        "whole song {seconds|round(digits=3)}s: notes view {total|round(digits=3)}s, of which the image"
        " {image|round(digits=3)}s; {rectangles} held rectangles ({per|round(digits=3)}/s), {keys|round(digits=3)}"
        " keys lit per frame; voice pass {voices} ({voice_per|round(digits=3)}/s), {voice_keys|round(digits=3)} keys",
        seconds=len(samples) / rate,
        total=time.monotonic() - imaged,
        image=imaged - start,
        rectangles=fields["rectangles"],
        per=fields["rectangles"] * rate / len(samples),
        keys=(held > 0).sum(axis=1).mean(),
        voices=fields["voice"],
        voice_per=fields["voice"] * rate / len(samples),
        voice_keys=(voice > 0).sum(axis=1).mean(),
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
