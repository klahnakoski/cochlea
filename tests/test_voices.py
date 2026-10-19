import numpy

from cochlea.harmonics import suppress
from cochlea.spectrogram import BINS, FRAME_RATE, FREQS, magnitude, normalize
from cochlea.voices import top

RATE = 44100
SECONDS = 12
PARTIALS = numpy.arange(1, 9)
BASS_LINE = [12, 12, 19, 17, 12, 12, 21, 19] * 3  # C2.., half a second each
CHORDS = [(36, 40, 43), (41, 45, 48), (43, 47, 50), (36, 40, 43)] * 2  # two seconds each, from 4 s
MELODY = [60, 62, 64, 67, 65, 64, 62, 60] * 4  # C6.., three eighths of a second each, from 0.25 s


def tone(bin, start, stop, gain, decay=0.0, vibrato=0.0):
    """A harmonic tone at `bin`, partial h at 1/h; decay in seconds (0 holds); vibrato in semitones at 5.5 Hz."""
    t = numpy.arange(SECONDS * RATE) / RATE
    pitch = FREQS[bin] * 2 ** (vibrato / 12 * numpy.sin(2 * numpy.pi * 5.5 * t))
    phase = 2 * numpy.pi * numpy.cumsum(pitch) / RATE
    gate = numpy.clip((t - start) / 0.01, 0, 1) * numpy.clip((stop - t) / 0.03, 0, 1)
    envelope = gate * (numpy.exp(-numpy.maximum(t - start, 0) / decay) if decay else 1)
    return gain * envelope * (numpy.sin(phase[:, None] * PARTIALS) / PARTIALS).sum(axis=1)


def kick(start):
    t = numpy.arange(SECONDS * RATE) / RATE - start
    phase = 2 * numpy.pi * (50 * t + 3 * (1 - numpy.exp(-numpy.maximum(t, 0) / 0.03)))
    return (t >= 0) * numpy.sin(phase) * numpy.exp(-numpy.maximum(t, 0) / 0.1)


def band():
    """Return (samples, truth): truth is (instruments, frames, BINS) bool, bass, pad and melody notes sounding."""
    notes = (
        [
            dict(instrument=0, bin=b, start=0.5 * i, stop=0.5 * i + 0.45, gain=1.0, decay=0.6)
            for i, b in enumerate(BASS_LINE)
        ]
        + [
            dict(instrument=1, bin=b, start=4 + 2 * i, stop=4 + 2 * i + 1.95, gain=0.25)
            for i, chord in enumerate(CHORDS[:4])
            for b in chord
        ]
        + [
            dict(instrument=2, bin=b, start=0.25 + 0.375 * i, stop=0.25 + 0.375 * i + 0.35, gain=0.5, vibrato=0.3)
            for i, b in enumerate(MELODY[:30])
        ]
    )
    samples = sum(tone(**{k: v for k, v in n.items() if k != "instrument"}) for n in notes)
    samples = samples + sum(0.8 * kick(0.5 * i) for i in range(2 * SECONDS))
    frames = -(-len(samples) // round(RATE / FRAME_RATE))
    times = numpy.arange(frames) / FRAME_RATE
    truth = numpy.zeros((3, frames, BINS), dtype=bool)
    for n in notes:
        truth[n["instrument"], (times > n["start"] + 0.05) & (times < n["stop"] - 0.05), n["bin"]] = True
    return (0.3 * samples / numpy.abs(samples).max()).astype("float32"), truth


def top_notes(samples):
    """Top notes over suppressed bars, as /analyze?view=top&suppress=1 computes them."""
    _, linear = magnitude(samples, RATE)
    return top(normalize(suppress(linear, samples=samples, sample_rate=RATE)), samples=samples, sample_rate=RATE)


def test_three_instruments_each_followed():
    samples, truth = band()
    lit, voices = top_notes(samples)
    assert 3 <= voices <= 5
    assert ((lit > 0).sum(axis=1) <= voices).all()
    sounding = truth.any(axis=2)
    found = (truth & (lit > 0)).any(axis=2)
    assert (found.sum(axis=1) / sounding.sum(axis=1) > 0.7).all()
