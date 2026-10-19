import numpy

from cochlea.notes import notes, stages
from cochlea.spectrogram import BINS, FRAME_RATE, FREQS

RATE = 44100
SECONDS = 4
A3, A4 = 33, 45
C3 = 24
PARTIALS = numpy.arange(1, 7)
TIME = numpy.arange(SECONDS * RATE) / RATE
HISS = 0.05 * numpy.random.default_rng(1).standard_normal(len(TIME))  # a noise floor, as under a song


def tone(bin, start, stop, partials=PARTIALS[:1]):
    """A tone at `bin` from `start` to `stop` seconds, partial h at 1/h."""
    gate = numpy.clip((TIME - start) / 0.01, 0, 1) * numpy.clip((stop - TIME) / 0.01, 0, 1)
    return 0.3 * gate * (numpy.sin(2 * numpy.pi * FREQS[bin] * partials * TIME[:, None]) / partials).sum(axis=1)


def kicks(every):
    """A kick with a broadband click every `every` seconds."""
    since = TIME % every
    body = numpy.sin(2 * numpy.pi * 50 * since) * numpy.exp(-since / 0.1)
    click = numpy.random.default_rng(0).standard_normal(len(TIME)) * numpy.exp(-since / 0.01)
    return body + click


def lit(samples):
    """(frames, BINS) Notes view grids: held rectangles, then the voice pass."""
    frames = -(-len(samples) // round(RATE / FRAME_RATE))
    grids, _ = notes(numpy.zeros((frames, BINS)), samples=samples.astype("float32"), sample_rate=RATE)
    return grids[:frames], grids[frames:]


def runs(column):
    """Number of separate lit stretches in one key's column."""
    on = numpy.concatenate([[False], column > 0])
    return int((on[1:] & ~on[:-1]).sum())


def test_held_tone_runs_through_kicks():
    frames, _ = lit(tone(A4, start=0.25, stop=3.75) + kicks(every=0.5))
    held = frames[round(0.5 * FRAME_RATE) : round(3.5 * FRAME_RATE), A4]
    assert (held > 0).all()
    assert runs(frames[:, A4]) == 1


def test_harmonic_tone_folds_to_its_fundamental():
    frames, _ = lit(tone(C3, start=0.25, stop=3.75, partials=PARTIALS))
    middle = frames[round(1 * FRAME_RATE) : round(3 * FRAME_RATE)]
    assert numpy.flatnonzero((middle > 0).any(axis=0)).tolist() == [C3]


def test_independent_octave_keeps_both():
    frames, _ = lit(tone(A3, start=0.25, stop=2.75) + tone(A4, start=1.25, stop=3.75))
    both = frames[round(1.5 * FRAME_RATE) : round(2.5 * FRAME_RATE)]
    assert (both[:, A3] > 0).all() and (both[:, A4] > 0).all()


def glide(first, last, start, stop, loudness=0.3):
    """A pure tone sliding evenly in pitch from bin `first` to `last` between `start` and `stop` seconds."""
    semitone = numpy.interp(TIME, [start, stop], [first, last])
    phase = 2 * numpy.pi * numpy.cumsum(FREQS[0] * 2 ** (semitone / 12)) / RATE
    gate = numpy.clip((TIME - start) / 0.01, 0, 1) * numpy.clip((stop - TIME) / 0.01, 0, 1)
    return loudness * gate * numpy.sin(phase)


def test_glide_is_one_stroke():
    samples = glide(A4, A4 + 4, start=1.5, stop=1.75) + HISS
    frames = -(-len(samples) // round(RATE / FRAME_RATE))
    rectangles = stages(samples.astype("float32"), sample_rate=RATE, frames=frames)["rectangles"]
    sliding = (rectangles["key"] >= A4) & (rectangles["key"] <= A4 + 4)
    assert sorted(rectangles["key"][sliding].tolist()) == list(range(A4, A4 + 5))
    assert numpy.unique(rectangles["stroke"][sliding]).size == 1


def test_quiet_vibrato_shows_in_the_voice_pass():
    vibrato = 0.4 * numpy.sin(2 * numpy.pi * 5.5 * TIME)
    voice = 0.3 * 10 ** (-38 / 20) * numpy.sin(2 * numpy.pi * FREQS[A4] * numpy.cumsum(2 ** (vibrato / 12)) / RATE)
    gate = numpy.clip((TIME - 1) / 0.01, 0, 1) * numpy.clip((3 - TIME) / 0.01, 0, 1)
    held, voiced = lit(tone(C3, start=0.25, stop=3.75) + gate * voice + HISS)
    singing = slice(round(1.25 * FRAME_RATE), round(2.75 * FRAME_RATE))
    assert (held[singing, A4] == 0).all()
    assert (voiced[singing, A4] > 0).mean() > 0.8
    assert (held[singing, C3] > 0).all()
