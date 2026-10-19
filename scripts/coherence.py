"""Fold bars whose envelopes move in unison: compare suppress with and without modulation coherence.

Run from the repo root: python -m scripts.coherence [song.wav]
"""

import sys
import time

import numpy
import soundfile
from mo_logs import logger

from cochlea.harmonics import (
    AUDIBLE,
    CAP,
    CLAIMABLE,
    FUNDAMENTALS,
    GATED,
    HARMONIC,
    OFFSETS,
    PRODUCT_OFFSETS,
    SALIENCE_FLOOR,
    above,
    attribute,
    below,
    harmonicity,
    pick,
    suppress,
)
from cochlea.spectrogram import BINS_PER_OCTAVE, FRAME_RATE, TAUS, magnitude, normalize
from scripts.harmonics import LIT, RATE, score, smooth
from scripts.instruments import CASES as INSTRUMENT_CASES, kicked

FAST = 3  # frames: modulation above ~20 Hz smoothed away
SLOW = 24  # frames: modulation below ~2.5 Hz removed
WINDOW = 180  # frames: three seconds of evidence
COHERENT = 0.8
CLIP = 0.1  # display level, 4.5 dB: an onset counts no more than a vibrato swing
STILL = 0.003  # display level, 0.14 dB: rms motion below this is a steady tone
C3, C4 = 36, 48
LAG = numpy.round(3 * TAUS * FRAME_RATE).astype(int)  # frames: the gammatone envelope peaks 3 tau late
VOICE = numpy.array([0.3, 1, 1, 0.8, 0.4, 0.2, 0.1, 0.05, 0.03, 0.02])


def motion(level):
    """(frames, bins): each bar's display level band-passed to 2..20 Hz, minus the frame's common motion."""
    level = level[(numpy.arange(len(level))[:, None] + LAG).clip(max=len(level) - 1), numpy.arange(level.shape[1])]
    moving = smooth(level, FAST) - smooth(level, SLOW)
    active = level > 0
    common = (moving * active).sum(axis=1, keepdims=True) / numpy.maximum(active.sum(axis=1, keepdims=True), 1)
    return ((moving - common).clip(-CLIP, CLIP) * active).astype("float32")


def sounding(level):
    """(frames, bins): no rise from silence nearby, so notes struck together are not one note."""
    return smooth((level <= 0).astype("float32"), SLOW + 2 * FAST) < 0.001


def coherence(moving, window=WINDOW):
    """(frames, bins, offsets): correlation of each bar's motion with the bar each offset below, over `window`."""
    power = smooth(moving**2, window)
    shared = smooth(moving[:, :, None] * below(moving, OFFSETS), window)
    return shared / numpy.sqrt((power[:, :, None] + STILL**2) * (below(power, OFFSETS) + STILL**2))


def folded(linear, samples, sample_rate):
    """suppress, plus: a bar coherent with an audible bar an overtone offset below hands it everything."""
    salience = numpy.exp(
        numpy.log(above(linear, PRODUCT_OFFSETS, floor=linear.max() * SALIENCE_FLOOR + 1e-30)).mean(axis=2)
    )
    harmonic = numpy.zeros_like(linear)
    harmonic[:, GATED:] = harmonicity(samples, sample_rate, frames=len(linear))
    pitched = (below(harmonic, OFFSETS) > HARMONIC) | (FUNDAMENTALS < GATED)
    fundamentals = below(linear, OFFSETS)
    audible = (fundamentals > AUDIBLE * linear[:, :, None]) * CLAIMABLE
    rivals = below(salience, OFFSETS) * audible * pitched
    choice = rivals.argmax(axis=2)
    beaten = pick(rivals, choice) > salience
    level = normalize(linear)
    bond = coherence(motion(level) * sounding(level)) * audible
    partner = bond.argmax(axis=2)
    coherent = pick(bond, partner) > COHERENT
    moved = numpy.where(
        coherent, linear, numpy.where(beaten, numpy.minimum(linear, CAP * pick(fundamentals, choice)), 0)
    )
    return attribute(linear, moved=moved, offset=OFFSETS[numpy.where(coherent, partner, choice)])


CASES = {
    **INSTRUMENT_CASES,
    "voice C4, vibrato": ([dict(bin=C4, start=0.2, stop=2.5, decay=0, timbre=VOICE, vibrato=0.3)], 0),
    "voice C3 + voice C4, own vibrato": (
        [
            dict(bin=C3, start=0.2, stop=2.5, decay=0, timbre=VOICE, vibrato=0.3, rate=5.0),
            dict(bin=C4, start=0.2, stop=2.5, decay=0, timbre=VOICE, vibrato=0.3, rate=6.7),
        ],
        0,
    ),
}
METHODS = {
    "suppress": lambda linear, samples, rate: suppress(linear, samples=samples, sample_rate=rate),
    "coherence": lambda linear, samples, rate: folded(linear, samples=samples, sample_rate=rate),
}


def unison(frames, lo=24, hi=72):
    """(lit bars per frame, mean coherence of lit bar pairs at overtone offsets), within C3..C7."""
    lit = frames > LIT
    span = lit[:, lo:hi]
    pairs = (lit[:, :, None] & below(lit, OFFSETS))[:, lo:hi]
    bond = coherence(motion(frames))[:, lo:hi]
    return span.sum(axis=1).mean(), bond[pairs].mean() if pairs.any() else 0.0


def compare():
    rows = []
    for name, (notes, loudness) in CASES.items():
        samples, truth = kicked(notes, loudness)
        _, linear = magnitude(samples, RATE)
        for method, shape in METHODS.items():
            frames = normalize(shape(linear, samples, RATE))
            recall, false, level = score(frames, truth)
            fundamentals = truth.any(axis=0).nonzero()[0]
            low = frames[truth.any(axis=1)][:, fundamentals.min()].mean()
            rows.append(
                f"{name:34} {method:10} recall {recall:5.2f}  false bars {false:5.2f}  note level {level:4.2f}"
                f"  lowest note {low:4.2f}"
            )
    print("\n".join(rows))


def octave_levels(frames):
    return [round(float(level), 2) for level in frames.reshape(len(frames), -1, BINS_PER_OCTAVE).mean(axis=(0, 2))]


def song(path):
    samples, rate = soundfile.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    _, linear = magnitude(mono, rate)
    for method, shape in METHODS.items():
        start = time.monotonic()
        frames = normalize(shape(linear, mono, rate))
        seconds = time.monotonic() - start
        lit, bond = unison(frames)
        print(f"{method:10} {octave_levels(frames)}  lit C3..C7 {lit:5.2f}  unison {bond:5.3f}  {seconds:.1f}s")


def main():
    logger.start()
    if len(sys.argv) > 1:
        song(sys.argv[1])
    else:
        compare()


if __name__ == "__main__":
    main()
