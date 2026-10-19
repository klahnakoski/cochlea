import numpy

from cochlea.notes import notes
from cochlea.spectrogram import BINS, FRAME_RATE, FREQS

RATE = 44100
SECONDS = 4
A3, A4 = 33, 45
C3 = 24
PARTIALS = numpy.arange(1, 7)
TIME = numpy.arange(SECONDS * RATE) / RATE


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
    """(frames, BINS) Notes view and its rectangle count."""
    frames = -(-len(samples) // round(RATE / FRAME_RATE))
    return notes(numpy.zeros((frames, BINS)), samples=samples.astype("float32"), sample_rate=RATE)


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
