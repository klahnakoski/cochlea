import numpy

from cochlea.spectrogram import BINS, FLOOR_DB, FMIN, bars, emphasize, magnitude, spectrogram

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


def test_bar_lights_at_attack_and_fades_after():
    for freq, early, late in ((55, 0.35, 1.0), (262, 0.1, 0.5)):  # every bin is read 3τ ahead and rings 8τ
        samples = numpy.zeros(4 * RATE, dtype="float32")
        samples[RATE : 2 * RATE] = numpy.sin(2 * numpy.pi * freq * numpy.arange(RATE) / RATE)
        hop, frames = spectrogram(samples, RATE)
        start, stop = RATE // hop, 2 * RATE // hop
        assert frames[: start - round(early * RATE / hop), bin_of(freq)].max() == 0, freq
        assert frames[stop + round(late * RATE / hop) :, bin_of(freq)].max() == 0, freq


def test_bass_notes_light_their_bar_after_each_attack():
    notes = ((65.406, 0.5), (73.416, 2.0))  # C2 then D2
    samples = numpy.zeros(4 * RATE, dtype="float32")
    t = numpy.arange(int(1.3 * RATE)) / RATE
    for freq, start in notes:
        samples[int(start * RATE) :][: len(t)] = numpy.sin(2 * numpy.pi * freq * t)
    hop, frames = spectrogram(samples, RATE)
    for freq, start in notes:
        assert frames[round((start + 0.5) * RATE / hop)].argmax() == bin_of(freq), freq


def test_equal_tones_follow_equal_loudness():
    c2, c3, c4, c5, c6 = 65.406, 130.81, 261.63, 523.25, 1046.5
    _, frames = spectrogram(tone(c2, c3, c4, c5, c6), RATE)
    db = frames[len(frames) // 2] * FLOOR_DB
    below_c3 = db[bin_of(c3)] - db[bin_of(c2)]
    low_mid = db[bin_of(c4)] - db[bin_of(c3)]
    high_mid = db[bin_of(c6)] - db[bin_of(c5)]
    assert abs(below_c3) < 1 and low_mid > 3 and abs(high_mid) < 2


def test_bass_note_outshines_its_octave():
    c1 = 32.703
    samples = tone(c1) + 0.5 * tone(2 * c1)  # 2nd harmonic 6 dB down
    _, frames = spectrogram(samples, RATE)
    middle = frames[len(frames) // 2]
    assert middle[bin_of(c1)] > middle[bin_of(2 * c1)]


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


def test_emphasis_tilts_6_db_per_octave():
    low, high = 110.0, 440.0
    emphasized = emphasize(tone(low) + tone(high))
    _, frames = spectrogram(emphasized, RATE)
    plain = spectrogram(tone(low) + tone(high), RATE)[1]
    middle, before = frames[len(frames) // 2] * FLOOR_DB, plain[len(plain) // 2] * FLOOR_DB
    tilt = (middle[bin_of(high)] - middle[bin_of(low)]) - (before[bin_of(high)] - before[bin_of(low)])
    assert abs(tilt - 12) < 1
