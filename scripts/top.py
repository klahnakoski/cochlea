"""Measure the Top notes view on the synthetic band (tests/test_voices.py) or a song.

Run from the repo root: python -m scripts.top [song.wav]
"""

import sys
import time

import numpy
import soundfile
from mo_logs import logger

from cochlea.harmonics import suppress
from cochlea.spectrogram import magnitude, normalize
from cochlea.voices import follow
from tests.test_voices import RATE, band

SHAPES = {"plain": lambda linear, samples, sample_rate: linear, "suppressed": suppress}


def measure(samples, sample_rate, shape):
    """Return (keys, voices, seconds past the shaped spectrogram)."""
    _, linear = magnitude(samples, sample_rate)
    levels = normalize(shape(linear, samples=samples, sample_rate=sample_rate))
    start = time.monotonic()
    keys, voices = follow(levels, samples, sample_rate)
    return keys, voices, time.monotonic() - start


def changes(keys):
    """Key changes of each voice: entering, leaving or moving."""
    return (keys[1:] != keys[:-1]).sum(axis=0)


def synthetic(name, shape):
    samples, truth = band()
    keys, voices, seconds = measure(samples, RATE, shape)
    lit = numpy.zeros(truth.shape[1:], dtype=bool)
    frame = numpy.broadcast_to(numpy.arange(len(keys))[:, None], keys.shape)
    lit[frame[keys >= 0], keys[keys >= 0]] = True
    any_truth = truth.any(axis=0)
    sounding = truth.any(axis=2)
    found = (truth & lit).any(axis=2)
    notes = (numpy.diff(truth.astype(int), axis=1) == 1).sum()
    logger.info(
        "synthetic {name}: {voices} voices; precision {p|round(places=2)}, recall {r|round(places=2)} of all notes;"
        " per instrument (bass, pad, melody) {per}; key changes {k} for {n} true notes; {s|round(places=2)}s",
        name=name,
        voices=voices,
        p=(lit & any_truth).sum() / max(lit.sum(), 1),
        r=(lit & any_truth).sum() / any_truth.sum(),
        per=(found.sum(axis=1) / sounding.sum(axis=1)).round(2).tolist(),
        k=changes(keys).tolist(),
        n=notes,
        s=seconds,
    )


def song(path, name, shape):
    samples, rate = soundfile.read(path, dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    keys, voices, seconds = measure(samples, rate, shape)
    duration = len(samples) / rate
    logger.info(
        "song {name}: {voices} voices; mean lit keys per frame {lit|round(places=2)}; key changes per second"
        " {k|round(places=1)}; {s|round(places=2)}s past the spectrogram for {d|round}s",
        name=name,
        voices=voices,
        lit=(keys >= 0).sum(axis=1).mean(),
        k=changes(keys).sum() / duration,
        s=seconds,
        d=duration,
    )


if __name__ == "__main__":
    logger.start()
    for name, shape in SHAPES.items():
        if len(sys.argv) > 1:
            song(sys.argv[1], name=name, shape=shape)
        else:
            synthetic(name, shape=shape)
