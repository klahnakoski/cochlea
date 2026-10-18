import numpy

from cochlea.harmonics import suppress
from cochlea.spectrogram import FREQS, magnitude, normalize

RATE = 44100
C3, C4, G4 = 36, 48, 55


def note(bin, start, decay, seconds=3):
    """Harmonic tone at `bin` from `start`: partial h at 1/h, higher partials decaying faster."""
    t = numpy.arange(seconds * RATE) / RATE
    age = numpy.maximum(t - start, 0)
    partials = numpy.arange(1, 7)
    waves = numpy.sin(2 * numpy.pi * FREQS[bin] * partials * t[:, None]) / partials
    return ((t >= start)[:, None] * numpy.exp(-age[:, None] * numpy.sqrt(partials) / decay) * waves).sum(axis=1)


def levels(samples, seconds, shape):
    """Bar levels 0..1 of C3, C4, G4 at `seconds`."""
    _, linear = magnitude(samples.astype("float32"), RATE)
    return normalize(shape(linear))[round(seconds * 60), [C3, C4, G4]]


def test_overtones_drop_fundamental_stays():
    tone = note(C3, start=0.2, decay=2)
    assert (levels(tone, seconds=1, shape=lambda linear: linear) > 0.5).all()
    fundamental, octave, twelfth = levels(tone, seconds=1, shape=suppress)
    assert fundamental > 0.9 and octave < 0.3 and twelfth < 0.3


def test_octave_of_separate_notes_keeps_both():
    octave = note(C3, start=0.2, decay=1) + note(C4, start=1, decay=1)
    low, high, _ = levels(octave, seconds=1.3, shape=suppress)
    assert low > 0.5 and high > 0.5


def test_bass_claims_only_its_first_overtone():
    c1, c2, g2 = 0, 12, 19
    _, linear = magnitude(note(c1, start=0.2, decay=2).astype("float32"), RATE)
    plain, suppressed = normalize(linear)[60], normalize(suppress(linear))[60]
    assert plain[c2] > 0.5 and plain[g2] > 0.5
    assert suppressed[c2] < 0.3 and suppressed[g2] > 0.5
