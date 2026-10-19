import numpy

from cochlea.spectrogram import BINS, FLOOR_DB, FMIN, bars, magnitude, spectrogram

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


def test_bass_burst_stays_short():
    samples = numpy.zeros(2 * RATE, dtype="float32")
    burst = int(0.05 * RATE)
    samples[RATE : RATE + burst] = numpy.sin(2 * numpy.pi * 55 * numpy.arange(burst) / RATE)
    hop, frames = spectrogram(samples, RATE)
    lit = numpy.flatnonzero(frames[:, bin_of(55)] > 0.5)
    assert lit.min() >= RATE // hop
    assert (lit.max() - lit.min()) * hop / RATE < 0.35


def test_bar_lights_at_attack_and_fades_after():
    for freq in (55, 262):
        samples = numpy.zeros(3 * RATE, dtype="float32")
        samples[RATE : 2 * RATE] = numpy.sin(2 * numpy.pi * freq * numpy.arange(RATE) / RATE)
        hop, frames = spectrogram(samples, RATE)
        start, stop = RATE // hop, 2 * RATE // hop
        assert frames[:start, bin_of(freq)].max() == 0, freq
        assert frames[stop + round(0.5 * RATE / hop) :, bin_of(freq)].max() == 0, freq


def test_equal_tones_follow_equal_loudness():
    c2, c3, c5, c6 = 65.406, 130.81, 523.25, 1046.5
    _, frames = spectrogram(tone(c2, c3, c5, c6), RATE)
    db = frames[len(frames) // 2] * FLOOR_DB
    bass_step = db[bin_of(c3)] - db[bin_of(c2)]
    mid_step = db[bin_of(c6)] - db[bin_of(c5)]
    assert bass_step > 5 and abs(mid_step) < 2


def test_finer_grid_centres_each_bar_and_rejects_neighbours_alike():
    _, coarse = magnitude(tone(440), RATE)
    _, fine = magnitude(tone(440), RATE, bins_per_octave=36)
    middle = fine[len(fine) // 2]
    assert fine.shape[1] == 3 * BINS
    assert middle.argmax() == 3 * 45 + 1
    fine_neighbour = middle[3 * 45 + 2] / middle.max()
    coarse_neighbour = coarse[len(coarse) // 2, 46] / coarse[len(coarse) // 2, 45]
    assert abs(fine_neighbour - coarse_neighbour) < 0.2 * coarse_neighbour
    assert bars(fine, 36)[len(fine) // 2].argmax() == 45
