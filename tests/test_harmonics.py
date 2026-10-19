import numpy

from cochlea.harmonics import suppress
from cochlea.spectrogram import FREQS, magnitude, normalize

RATE = 44100
C4, C5, G5 = 36, 48, 55
C3, C7 = 24, 72


def note(bin, start, decay, seconds=3):
    """Harmonic tone at `bin` from `start`: partial h at 1/h, higher partials decaying faster."""
    t = numpy.arange(seconds * RATE) / RATE
    age = numpy.maximum(t - start, 0)
    partials = numpy.arange(1, 7)
    waves = numpy.sin(2 * numpy.pi * FREQS[bin] * partials * t[:, None]) / partials
    return ((t >= start)[:, None] * numpy.exp(-age[:, None] * numpy.sqrt(partials) / decay) * waves).sum(axis=1)


def pink_noise(seconds=3):
    """Noise falling 3 dB per octave, like a busy mix."""
    spectrum = numpy.fft.rfft(numpy.random.default_rng(0).standard_normal(seconds * RATE))
    return numpy.fft.irfft(spectrum / numpy.sqrt(numpy.arange(len(spectrum)) + 1), n=seconds * RATE)


def suppressed(samples):
    """(plain, suppressed) bar levels 0..1, every frame."""
    samples = samples.astype("float32")
    _, linear = magnitude(samples, RATE)
    return normalize(linear), normalize(suppress(linear, samples=samples, sample_rate=RATE))


def test_overtones_drop_fundamental_stays():
    plain, shaped = suppressed(note(C4, start=0.2, decay=2))
    assert (plain[60, [C4, C5, G5]] > 0.5).all()
    fundamental, octave, twelfth = shaped[60, [C4, C5, G5]]
    assert fundamental > 0.9 and octave < 0.3 and twelfth < 0.3


def test_octave_of_separate_notes_keeps_both():
    _, shaped = suppressed(note(C4, start=0.2, decay=1) + note(C5, start=1, decay=1))
    low, high = shaped[78, [C4, C5]]
    assert low > 0.5 and high > 0.5


def test_bass_claims_only_its_first_overtone():
    c1, c2, g2 = 0, 12, 19
    plain, shaped = suppressed(note(c1, start=0.2, decay=2))
    assert plain[60, c2] > 0.5 and plain[60, g2] > 0.5
    assert shaped[60, c2] < 0.3 and shaped[60, g2] > 0.5


def test_noise_is_no_ones_overtone():
    plain, shaped = suppressed(pink_noise())
    assert shaped[30:, C3:C7].mean() > 0.9 * plain[30:, C3:C7].mean()
