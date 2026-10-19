"""Learn instruments (one overtone curve each, at any pitch) with shift-invariant NMF.

Run from the repo root: python -m scripts.instruments [song.wav]
"""

import sys
import time

import numpy
import soundfile
from mo_logs import logger

from cochlea.harmonics import suppress
from cochlea.spectrogram import BINS, BINS_PER_OCTAVE, magnitude, normalize
from scripts.harmonics import PARTIALS, RATE, score, signal

SPAN = 48  # overtone curve covers offsets 0..47 semitones
HARMONICS = numpy.round(12 * numpy.log2(numpy.arange(1, 16))).astype(int)  # offsets of partials 1..15
INSTRUMENTS = 6
EPS = 1e-12


def shifts():
    """(BINS pitches, BINS bins) offset b - p, and whether it lies on the curve."""
    offset = numpy.arange(BINS)[None, :] - numpy.arange(BINS)[:, None]
    return offset.clip(0, SPAN - 1), (offset >= 0) & (offset < SPAN)


OFFSET, ON_CURVE = shifts()
ALONG = numpy.arange(BINS)[:, None] + numpy.arange(SPAN)  # (pitch, offset) -> bin
INSIDE = ALONG < BINS


def dictionary(curves):
    """(instruments * pitches, bins): every curve placed at every pitch."""
    return (curves[:, OFFSET] * ON_CURVE).reshape(-1, BINS)


def initial_curves(count):
    """Harmonic curves from bright (1/h^0.5) to mellow (1/h^2.5); off-harmonic offsets small."""
    curves = numpy.full((count, SPAN), 0.01)
    rolloff = numpy.linspace(0.5, 2.5, count)
    numpy.maximum.at(curves, (slice(None), HARMONICS), numpy.arange(1.0, 16) ** -rolloff[:, None])
    return curves


def update_activations(spectra, atoms, activations, sparsity):
    """One KL multiplicative update of activations (frames, atoms)."""
    ratio = spectra / (activations @ atoms + EPS)
    return activations * (ratio @ atoms.T) / (atoms.sum(axis=1) + sparsity)


def update_curves(spectra, curves, activations):
    """One KL multiplicative update of curves, then peak-normalize each."""
    ratio = spectra / (activations @ dictionary(curves) + EPS)
    gradient = (activations.T @ ratio).reshape(len(curves), BINS, BINS)  # (instrument, pitch, bin)
    numerator = (gradient[:, numpy.arange(BINS)[:, None], ALONG.clip(max=BINS - 1)] * INSIDE).sum(axis=1)
    totals = activations.sum(axis=0).reshape(len(curves), BINS)
    denominator = totals @ INSIDE + EPS
    curves = curves * numerator / denominator
    return curves / curves.max(axis=1, keepdims=True)


def learn(
    linear, count=INSTRUMENTS, sparsity=0.3, note_sparsity=0.3, stride=4, curve_iterations=60, activation_iterations=40
):
    """Return (curves (count, SPAN), activations (frames, count * BINS)).

    KL multiplicative updates with an L1 prior on activations, started from suppress().
    Curves are learned on every `stride`th frame; activations are then fit on all frames.
    """
    spectra = linear / (linear.max() + EPS)
    start = numpy.tile(suppress(spectra), count) / count + 1e-3 * spectra.mean()  # notes at their fundamentals
    sample = spectra[::stride]
    curves = initial_curves(count)
    activations = start[::stride]
    for _ in range(curve_iterations):
        activations = update_activations(sample, dictionary(curves), activations, sparsity)
        curves = update_curves(sample, curves, activations)
    atoms = dictionary(curves)
    activations = start
    for _ in range(activation_iterations):
        activations = update_activations(spectra, atoms, activations, note_sparsity)
    return curves, activations * linear.max()


def pitches(linear, curves, activations):
    """(frames, BINS): each pitch's share of the measured magnitude, summed over instruments.

    A bin's magnitude is split among the notes in proportion to the model (Wiener mask);
    a note's magnitude is the root of its share's power over all its bins.
    """
    atoms = dictionary(curves)
    ratio = linear / (activations @ atoms + EPS)
    notes = activations * numpy.sqrt(ratio**2 @ (atoms**2).T)
    return notes.reshape(len(linear), len(curves), BINS).sum(axis=1).astype("float32")


def instruments(linear, **kwargs):
    return pitches(linear, *learn(linear, **kwargs))


C3, C4, E4, G4, C5 = 36, 48, 52, 55, 60
BRIGHT = 1 / PARTIALS  # saw-like
MELLOW = numpy.array([1, 0.3, 0.1, 0, 0, 0, 0, 0, 0, 0])


def melody(bins, timbre, start, step, decay=0.8):
    return [
        dict(bin=b, start=start + step * i, stop=start + step * (i + 1), decay=decay, seconds=6.0, timbre=timbre)
        for i, b in enumerate(bins)
    ]


CASES = {
    "bright melody": melody([C4, E4, G4, C5, G4, E4], BRIGHT, start=0.2, step=0.9),
    "mellow melody": melody([C4, E4, G4, C5, G4, E4], MELLOW, start=0.2, step=0.9),
    "bright bass + mellow melody": melody([C3, G4 - 24, C3, G4 - 24, C3], BRIGHT, start=0.2, step=1.1)
    + melody([E4, G4, C5, G4, E4, C4], MELLOW, start=0.5, step=0.9),
    "mellow bass + bright melody": melody([C3, G4 - 24, C3, G4 - 24, C3], MELLOW, start=0.2, step=1.1)
    + melody([E4, G4, C5, G4, E4, C4], BRIGHT, start=0.5, step=0.9),
    "octave: bright C3, mellow C4": melody([C3] * 5, BRIGHT, start=0.2, step=1.1)
    + melody([C4, C4, E4, C4, C4], MELLOW, start=0.6, step=1.0),
    "octave: mellow C3, bright C4": melody([C3] * 5, MELLOW, start=0.2, step=1.1)
    + melody([C4, C4, E4, C4, C4], BRIGHT, start=0.6, step=1.0),
}
METHODS = {
    "none": lambda linear: linear,
    "suppress": suppress,
    "nmf": instruments,
}


def compare():
    rows = []
    for name, notes in CASES.items():
        samples, truth = signal(notes)
        _, linear = magnitude(samples, RATE)
        for method, shape in METHODS.items():
            recall, false, level = score(normalize(shape(linear)), truth)
            rows.append(f"{name:30} {method:12} recall {recall:5.2f}  false bars {false:5.2f}  note level {level:4.2f}")
    print("\n".join(rows))


def octave_levels(frames):
    return frames.reshape(len(frames), -1, BINS_PER_OCTAVE).mean(axis=(0, 2)).round(2).tolist()


def song(path):
    samples, rate = soundfile.read(path, dtype="float32", always_2d=True)
    _, linear = magnitude(samples.mean(axis=1), rate)
    print("plain   ", octave_levels(normalize(linear)))
    print("suppress", octave_levels(normalize(suppress(linear))))
    start = time.monotonic()
    curves, activations = learn(linear)
    print(
        f"nmf      {octave_levels(normalize(pitches(linear, curves, activations)))}  in {time.monotonic() - start:.1f}s"
    )
    for i, curve in enumerate(curves):
        print(f"curve {i}: " + " ".join(f"{v:.2f}" for v in curve[:41]))


def main():
    logger.start()
    if len(sys.argv) > 1:
        song(sys.argv[1])
    else:
        compare()


if __name__ == "__main__":
    main()
