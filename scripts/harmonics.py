"""Compare harmonic-suppression methods on synthesized instrument tones.

Run from the repo root: python -m scripts.harmonics
"""

import numpy
from mo_logs import logger

from cochlea.harmonics import AUDIBLE, OFFSETS, PRODUCT_OFFSETS, SALIENCE_FLOOR, above, attribute, below, pick, suppress
from cochlea.spectrogram import BINS, FRAME_RATE, FREQS, magnitude, normalize

RATE = 44100
PARTIALS = numpy.arange(1, 11)
LIT = 0.5


def note(bin, start, stop, decay, tremolo=0.0, vibrato=0.0, rate=5.5, seconds=3.0, timbre=1 / PARTIALS):
    """A harmonic tone at `bin`: partial h at timbre[h - 1], decaying faster as h rises; vibrato in semitones."""
    t = numpy.arange(int(seconds * RATE)) / RATE
    pitch = FREQS[bin] * 2 ** (vibrato / 12 * numpy.sin(2 * numpy.pi * rate * t))
    phase = 2 * numpy.pi * numpy.cumsum(pitch) / RATE
    age = t - start
    gate = numpy.clip(age / 0.01, 0, 1) * numpy.clip((stop - t) / 0.05, 0, 1)
    swell = 1 + tremolo * numpy.sin(2 * numpy.pi * 5 * t)
    decays = numpy.exp(-numpy.maximum(age, 0)[:, None] * numpy.sqrt(PARTIALS) / decay) if decay else 1
    partials = numpy.sin(phase[:, None] * PARTIALS) * timbre * decays
    return (gate * swell)[:, None] * partials


def signal(notes):
    """Return (samples, truth): truth is a (frames, BINS) bool grid of sounding notes."""
    samples = sum(note(**n).sum(axis=1) for n in notes)
    samples = (0.3 * samples / numpy.abs(samples).max()).astype("float32")
    frames = -(-len(samples) // round(RATE / FRAME_RATE))
    times = numpy.arange(frames) / FRAME_RATE
    truth = numpy.zeros((frames, BINS), dtype=bool)
    for n in notes:
        truth[(times > n["start"] + 0.03) & (times < n["stop"] - 0.05), n["bin"]] = True
    return samples, truth


def smooth(x, width):
    """Centred moving mean along time, same length, edges held."""
    half = width // 2
    padded = numpy.pad(x, [(half, width - 1 - half)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    total = numpy.concatenate([numpy.zeros_like(padded[:1]), numpy.cumsum(padded, axis=0)])
    return (total[width:] - total[:-width]) / width


def hps(linear, allowed=True):
    """Uncapped harmonic product spectrum: an overtone hands over all its energy."""
    salience = numpy.exp(numpy.log(above(linear, PRODUCT_OFFSETS, floor=linear.max() * SALIENCE_FLOOR)).mean(axis=2))
    rivals = below(salience, OFFSETS) * (below(linear, OFFSETS) > AUDIBLE * linear[:, :, None]) * allowed
    choice = rivals.argmax(axis=2)
    moved = numpy.where(pick(rivals, choice) > salience, linear, 0)
    return attribute(linear, moved=moved, offset=OFFSETS[choice])


def hybrid(linear, width=30, varies=0.1):
    """HPS, except where both envelopes vary but do not co-vary: then they are separate notes."""
    upper = linear.astype("float64")[:, :, None]
    lower = below(linear.astype("float64"), OFFSETS)
    mean_u, mean_l = smooth(upper, width), smooth(lower, width)
    cov = smooth(upper * lower, width) - mean_u * mean_l
    var_u = smooth(upper**2, width) - mean_u**2
    var_l = smooth(lower**2, width) - mean_l**2
    fit = numpy.maximum(cov, 0) ** 2 / (var_u * var_l + 1e-30)
    moving = (var_u > (varies * mean_u) ** 2) | (var_l > (varies * mean_l) ** 2)
    return hps(linear, allowed=~moving | (fit > 0.5))


def correlation(linear, width=30, slack=0.02):
    """A bin whose envelope co-varies with a bin an overtone interval below hands its energy down."""
    upper = linear.astype("float64")[:, :, None]
    lower = below(linear.astype("float64"), OFFSETS)
    mean_u, mean_l = smooth(upper, width), smooth(lower, width)
    cov = smooth(upper * lower, width) - mean_u * mean_l
    var_u = smooth(upper**2, width) - mean_u**2 + (slack * mean_u) ** 2
    var_l = smooth(lower**2, width) - mean_l**2 + (slack * mean_l) ** 2 + 1e-30
    fit = numpy.maximum(cov, 0) ** 2 / (var_u * var_l)
    gate = ((fit - 0.5) / 0.4).clip(0, 1) * (mean_l > AUDIBLE * mean_u)
    explained = gate * numpy.minimum(upper, mean_u / (mean_l + 1e-30) * lower)
    choice = explained.argmax(axis=2)
    return attribute(linear, moved=pick(explained, choice), offset=OFFSETS[choice])


C3, C4, E4, G4, C5 = 36, 48, 52, 55, 60
CASES = {
    "decaying C3": [dict(bin=C3, start=0.2, stop=2.5, decay=0.8)],
    "C3 saw, tremolo": [dict(bin=C3, start=0.2, stop=2.5, decay=0, tremolo=0.1)],
    "C3 saw, steady": [dict(bin=C3, start=0.2, stop=2.5, decay=0)],
    "C3 then C4": [dict(bin=C3, start=0.2, stop=2.5, decay=0.8), dict(bin=C4, start=0.8, stop=2.5, decay=0.8)],
    "C3 then G4": [dict(bin=C3, start=0.2, stop=2.5, decay=0.8), dict(bin=G4, start=0.8, stop=2.5, decay=0.8)],
    "C4 then C3": [dict(bin=C4, start=0.2, stop=2.5, decay=0.8), dict(bin=C3, start=0.8, stop=2.5, decay=0.8)],
    "C3+C4 struck together": [
        dict(bin=C3, start=0.2, stop=2.5, decay=0.8),
        dict(bin=C4, start=0.2, stop=2.5, decay=0.8),
    ],
    "C major chord": [dict(bin=b, start=0.2, stop=2.5, decay=0.8) for b in (C4, E4, G4)],
    "melody C4 E4 G4 C5": [
        dict(bin=b, start=0.2 + 0.4 * i, stop=1.7 + 0.4 * i, decay=0.6) for i, b in enumerate((C4, E4, G4, C5))
    ],
}
METHODS = {
    "none": lambda linear, samples: linear,
    "hps": lambda linear, samples: hps(linear),
    "correlation": lambda linear, samples: correlation(linear),
    "hps + corr veto": lambda linear, samples: hybrid(linear),
    "suppress": lambda linear, samples: suppress(linear, samples=samples, sample_rate=RATE),
}


def score(frames, truth):
    """(note recall, false bars per frame, mean note level) over frames with a note sounding."""
    active = truth.any(axis=1)
    lit = frames[active] > LIT
    notes = truth[active]
    return lit[notes].mean(), (lit & ~notes).sum(axis=1).mean(), frames[active][notes].mean()


def main():
    logger.start()
    rows = []
    for name, notes in CASES.items():
        samples, truth = signal(notes)
        _, linear = magnitude(samples, RATE)
        for method, shape in METHODS.items():
            recall, false, level = score(normalize(shape(linear, samples)), truth)
            rows.append(f"{name:24} {method:16} recall {recall:5.2f}  false bars {false:5.2f}  note level {level:4.2f}")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
