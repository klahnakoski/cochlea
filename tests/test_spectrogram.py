import numpy

from cochlea.spectrogram import BINS, FMIN, spectrogram

RATE = 44100


def tone(*freqs, seconds=2):
    t = numpy.arange(seconds * RATE) / RATE
    return sum(numpy.sin(2 * numpy.pi * f * t) for f in freqs).astype("float32")


def bin_of(freq):
    return round(12 * numpy.log2(freq / FMIN))


def lit_bins(samples):
    _, frames = spectrogram(samples, RATE)
    middle = frames[len(frames) // 2]
    return set(numpy.flatnonzero(middle > 0.8))


def test_shape_and_hop():
    hop, frames = spectrogram(tone(440), RATE)
    assert hop == 735
    assert frames.shape == (120, BINS)
    assert 0 <= frames.min() and frames.max() == 1


def test_a4_tone_lights_a4_bin():
    assert bin_of(440) == 45
    assert lit_bins(tone(440)) == {45}


def test_c_major_chord_lights_three_bins():
    chord = tone(261.6256, 329.6276, 391.9954)
    assert lit_bins(chord) == {bin_of(261.6256), bin_of(329.6276), bin_of(391.9954)}


def test_silence_is_zero():
    _, frames = spectrogram(numpy.zeros(RATE, dtype="float32"), RATE)
    assert not frames.any()
