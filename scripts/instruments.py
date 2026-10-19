"""Learn instruments (one overtone curve each, at any pitch) with shift-invariant NMF.

The fit runs on a grid finer than the display, so a peak's smear stays below its
harmonics; the result folds to the display's bars.

Run from the repo root: python -m scripts.instruments [song.wav]
"""

import sys
import time

import numpy
import soundfile
from mo_logs import logger

from cochlea.harmonics import CAP, suppress
from cochlea.loudness import gain_db
from cochlea.spectrogram import BINS_PER_OCTAVE, OCTAVES, PHON, bars, grid, magnitude, normalize
from scripts.harmonics import PARTIALS, RATE, score, signal

FIT = 36  # bins per octave of the fit grid
FIT_BINS = OCTAVES * FIT
SPAN = 4 * FIT  # overtone curve covers four octaves
HARMONICS = numpy.round(FIT * numpy.log2(numpy.arange(1, 16))).astype(int)  # offsets of partials 1..15
HEARD = 10 ** (gain_db(grid(FIT)[0], PHON) / 20)  # fit what the ear hears, so bass does not dominate
INSTRUMENTS = 6
EPS = 1e-12


def shifts():
    """(FIT_BINS pitches, FIT_BINS bins) offset b - p, and whether it lies on the curve."""
    offset = numpy.arange(FIT_BINS)[None, :] - numpy.arange(FIT_BINS)[:, None]
    return offset.clip(0, SPAN - 1), (offset >= 0) & (offset < SPAN)


OFFSET, ON_CURVE = shifts()
ALONG = numpy.arange(FIT_BINS)[:, None] + numpy.arange(SPAN)  # (pitch, offset) -> bin
INSIDE = ALONG < FIT_BINS


def dictionary(curves):
    """(instruments * pitches, bins): every curve placed at every pitch."""
    return (curves[:, OFFSET] * ON_CURVE).reshape(-1, FIT_BINS)


def initial_curves(count):
    """Harmonic curves from bright (1/h^0.5) to mellow (1/h^2.5); off-harmonic offsets small."""
    curves = numpy.full((count, SPAN), 0.01, dtype="float32")
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
    gradient = (activations.T @ ratio).reshape(len(curves), FIT_BINS, FIT_BINS)  # (instrument, pitch, bin)
    numerator = (gradient[:, numpy.arange(FIT_BINS)[:, None], ALONG.clip(max=FIT_BINS - 1)] * INSIDE).sum(axis=1)
    totals = activations.sum(axis=0).reshape(len(curves), FIT_BINS)
    denominator = totals @ INSIDE + EPS
    curves = curves * numerator / denominator
    return curves / curves.max(axis=1, keepdims=True)


def widen(spectra):
    """Max over each bin and its two neighbours: partials fall between bins."""
    padded = numpy.pad(spectra, [(0, 0), (1, 1)])
    return numpy.maximum(numpy.maximum(padded[:, :-2], padded[:, 1:-1]), padded[:, 2:])


def fundamentals(spectra, partials=5, floor=1e-3):
    """(frames, FIT_BINS): spectra, less CAP of the fundamental where a pitch an overtone below has more salience.

    Salience is the geometric mean of a pitch's first partials (harmonic product spectrum).
    """
    peaks = widen(spectra)
    index = numpy.arange(FIT_BINS)[:, None] + HARMONICS[:partials]
    above = numpy.where(index < FIT_BINS, peaks[:, index.clip(max=FIT_BINS - 1)], 0)
    salience = numpy.exp(numpy.log(numpy.maximum(above, floor)).mean(axis=2))
    index = numpy.arange(FIT_BINS)[:, None] - HARMONICS[1:partials]
    rivals = numpy.where(index >= 0, widen(salience)[:, index.clip(min=0)], 0)
    choice = rivals.argmax(axis=2)[:, :, None]
    beaten = numpy.take_along_axis(rivals, choice, axis=2)[:, :, 0] > salience
    fundamental = numpy.take_along_axis(peaks[:, index.clip(min=0)], choice, axis=2)[:, :, 0]
    return numpy.where(beaten, numpy.maximum(spectra - CAP * fundamental, 0), spectra)


def learn(
    linear, count=INSTRUMENTS, sparsity=0.3, note_sparsity=0.3, stride=4, curve_iterations=60, activation_iterations=40
):
    """Return (curves (count, SPAN), activations (frames, count * FIT_BINS)) for a FIT-grid magnitude.

    KL multiplicative updates with an L1 prior on activations, every pitch started at its bin.
    Curves are learned on every `stride`th frame; activations are then fit on all frames.
    """
    spectra = (linear / (linear.max() + EPS)).astype("float32")
    start = numpy.tile(fundamentals(spectra), count) / count
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
    """(frames, FIT_BINS): each pitch's share of the measured magnitude, summed over instruments.

    A bin's magnitude is split among the notes in proportion to the model (Wiener mask);
    a note's magnitude is the root of its share's power over all its bins.
    """
    atoms = dictionary(curves)
    ratio = linear / (activations @ atoms + EPS)
    notes = activations * numpy.sqrt(ratio**2 @ (atoms**2).T)
    return notes.reshape(len(linear), len(curves), FIT_BINS).sum(axis=1)


def instruments(linear, **kwargs):
    """(frames, BINS) note magnitudes from a FIT-grid magnitude."""
    heard = linear * HEARD
    return bars(pitches(heard, *learn(heard, **kwargs)) / HEARD, FIT).astype("float32")


def harmonic_levels(curve):
    """Curve value at partials 2..5."""
    return " ".join(f"{curve[h]:.2f}" for h in HARMONICS[1:5])


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
    "none": lambda display, fine: display,
    "suppress": lambda display, fine: suppress(display),
    f"nmf@{FIT}": lambda display, fine: instruments(fine),
}


def compare():
    rows = []
    for name, notes in CASES.items():
        samples, truth = signal(notes)
        _, display = magnitude(samples, RATE)
        _, fine = magnitude(samples, RATE, bins_per_octave=FIT)
        for method, shape in METHODS.items():
            recall, false, level = score(normalize(shape(display, fine)), truth)
            rows.append(f"{name:30} {method:12} recall {recall:5.2f}  false bars {false:5.2f}  note level {level:4.2f}")
    print("\n".join(rows))


def octave_levels(frames):
    return frames.reshape(len(frames), -1, BINS_PER_OCTAVE).mean(axis=(0, 2)).round(2).tolist()


def song(path):
    samples, rate = soundfile.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    _, display = magnitude(mono, rate)
    print("plain   ", octave_levels(normalize(display)))
    print("suppress", octave_levels(normalize(suppress(display))))
    start = time.monotonic()
    _, fine = magnitude(mono, rate, bins_per_octave=FIT)
    spectrogram_seconds = time.monotonic() - start
    print(f"plain@{FIT}", octave_levels(normalize(bars(fine, FIT))))
    start = time.monotonic()
    heard = fine * HEARD
    curves, activations = learn(heard)
    levels = octave_levels(normalize(bars(pitches(heard, curves, activations) / HEARD, FIT)))
    print(f"nmf@{FIT}  {levels}  spectrogram {spectrogram_seconds:.1f}s, nmf {time.monotonic() - start:.1f}s")
    for i, curve in enumerate(curves):
        print(f"curve {i}: harmonics 2..5 {harmonic_levels(curve)}")


def main():
    logger.start()
    if len(sys.argv) > 1:
        song(sys.argv[1])
    else:
        compare()


if __name__ == "__main__":
    main()
