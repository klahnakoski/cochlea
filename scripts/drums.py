"""Find repeated drum samples, subtract them, and measure what is left.

Run from the repo root: python -m scripts.drums [song.wav] [output folder]
With no song, a synthesized kick + snare + hat loop over a sustained melody.
"""

import sys
import time

import numpy
import soundfile
from mo_files import File
from mo_logs import logger

from cochlea.drums import separate
from cochlea.spectrogram import BINS_PER_OCTAVE, magnitude, normalize

RATE = 44100
OUTPUT = "C:/Users/kyle/AppData/Local/Temp/claude/C--Users-kyle-code-cochlea/895395c3-8fc2-43a7-b4db-3afa78ecedf7/scratchpad/drums"
TEMPO = 123  # beats per minute, so hits fall between samples
BARS = 12
TOLERANCE = 0.005  # seconds, a detected hit matches a true one
HIT_SECONDS = 0.3  # window after each hit where residual energy is measured


def kick(rate):
    t = numpy.arange(round(0.35 * rate)) / rate
    phase = 2 * numpy.pi * (50 * t + 100 * 0.03 * (1 - numpy.exp(-t / 0.03)))
    return numpy.sin(phase) * numpy.exp(-t / 0.12)


def snare(rate):
    t = numpy.arange(round(0.25 * rate)) / rate
    noise = numpy.random.default_rng(1).standard_normal(len(t))
    return 0.5 * noise * numpy.exp(-t / 0.06) + 0.6 * numpy.sin(2 * numpy.pi * 185 * t) * numpy.exp(-t / 0.05)


def hat(rate):
    t = numpy.arange(round(0.08 * rate)) / rate
    noise = numpy.diff(numpy.random.default_rng(2).standard_normal(len(t) + 2), n=2)
    return 0.15 * noise * numpy.exp(-t / 0.02)


def loop(rate):
    """(drums, truth): kick every beat, snare on 2 and 4, hats on eighths; truth is (time, drum) columns."""
    beat = 60 / TEMPO
    eighths = numpy.arange(BARS * 8) * beat / 2
    rng = numpy.random.default_rng(3)
    times = numpy.concatenate([eighths[::2], eighths[2::4], eighths])
    drum = numpy.repeat([0, 1, 2], [len(eighths[::2]), len(eighths[2::4]), len(eighths)])
    gain = numpy.select(
        [drum == 0, drum == 1], [1.0, rng.uniform(0.8, 1.0, len(drum))], rng.uniform(0.5, 1.0, len(drum))
    )
    start = numpy.round((times + 0.25) * rate).astype(int)
    samples = [kick(rate), snare(rate), hat(rate)]
    width = max(len(s) for s in samples)
    bank = numpy.stack([numpy.pad(s, (0, width - len(s))) for s in samples])
    out = numpy.zeros(start.max() + width + rate)
    numpy.add.at(out, start[:, None] + numpy.arange(width), gain[:, None] * bank[drum])
    return out, start, drum


def melody(length, rate):
    """Sustained notes, partials at 1/h, a new note every 1.5 s, gliding in over 50 ms."""
    t = numpy.arange(length) / rate
    notes = 261.63 * 2 ** (numpy.array([0, 4, 7, 12, 7, 4, 2, 5]) / 12)
    which = (t // 1.5).astype(int) % len(notes)
    age = t % 1.5
    partials = numpy.arange(1, 9)
    phase = 2 * numpy.pi * numpy.cumsum(notes[which]) / rate
    tone = (numpy.sin(phase[:, None] * partials) / partials).sum(axis=1)
    return tone * numpy.clip(age / 0.05, 0, 1) * numpy.clip((1.5 - age) / 0.05, 0, 1)


def snr(signal, noise):
    return 10 * numpy.log10((signal**2).sum() / (noise**2).sum())


def hit_windows(starts, length, rate):
    index = (starts[:, None] + numpy.arange(round(HIT_SECONDS * rate))).ravel()
    return numpy.unique(index[index < length])


def octave_levels(frames):
    return frames.reshape(len(frames), -1, BINS_PER_OCTAVE).mean(axis=(0, 2)).round(2).tolist()


def report(samples, rate, name):
    """Separate, print the kit and the residual numbers, write the WAVs; return (kit, residual)."""
    start = time.monotonic()
    kit = separate(samples, rate)
    seconds = time.monotonic() - start
    per_drum = kit.render(len(samples))
    drums = per_drum.sum(axis=0)
    residual = samples - drums
    total = (samples**2).sum()
    print(f"{name}: {len(samples) / rate:.0f} s of audio, separate {seconds:.1f} s")
    hits = numpy.bincount(kit.drum, minlength=len(kit.templates))
    for d in range(len(kit.templates)):  # report lines only
        mine = kit.drum == d
        print(
            f"  drum {d}: {kit.onset_hits[d]} onsets, patch similarity {kit.similarity[d]:.2f}, "
            f"{hits[d]} matched hits, score median {numpy.median(kit.score[mine]):.2f}, "
            f"gain median {numpy.median(kit.gain[mine]):.2f} [{kit.gain[mine].min():.2f}..{kit.gain[mine].max():.2f}], "
            f"energy {100 * (per_drum[d] ** 2).sum() / total:.1f}%"
        )
    print(
        f"  removed {100 * (1 - (residual**2).sum() / total):.1f}% of energy, drums alone {100 * (drums**2).sum() / total:.1f}%"
    )
    at = hit_windows(kit.time, len(samples), rate)
    print(
        f"  at hits: residual {10 * numpy.log10((residual[at] ** 2).sum() / (samples[at] ** 2).sum()):.1f} dB vs before"
    )
    _, before = magnitude(samples, rate)
    _, after = magnitude(residual, rate)
    both = normalize(numpy.concatenate([before, after]))  # residual on the original's scale
    print(f"  octave means C1..C8 before {octave_levels(both[: len(before)])}")
    print(f"  octave means C1..C8 after  {octave_levels(both[len(before) :])}")
    folder = File(OUTPUT)
    folder.create()
    soundfile.write((folder / f"{name} residual.wav").os_path, residual, rate)
    soundfile.write((folder / f"{name} drums.wav").os_path, drums, rate)
    return kit, residual


def synthetic():
    drums, start, drum = loop(RATE)
    tune = 0.5 * melody(len(drums), RATE) * numpy.sqrt((drums**2).mean() / melody(len(drums), RATE).var())
    samples = (drums + tune).astype("float32")
    kit, residual = report(samples, RATE, "synthetic")

    # each detected drum's constant offset from the sample start, then the spread
    events = numpy.unique(start)
    nearest = events[numpy.clip(numpy.searchsorted(events, kit.time), 1, len(events) - 1) - [[1], [0]]]
    error = kit.time - nearest[numpy.abs(kit.time - nearest).argmin(axis=0), numpy.arange(len(kit.time))]
    order = numpy.lexsort([error, kit.drum])
    first = numpy.searchsorted(kit.drum[order], numpy.arange(len(kit.templates)))
    count = numpy.bincount(kit.drum, minlength=len(kit.templates))
    offset = error[order][numpy.minimum(first + count // 2, len(order) - 1)]
    jitter = error - offset[kit.drum]
    matched = numpy.abs(jitter) <= TOLERANCE * RATE
    found = numpy.isin(events, kit.time[matched] - offset[kit.drum[matched]])
    print(
        f"  truth {len(start)} hits at {len(events)} event times; recall {found.mean():.3f}, "
        f"precision {matched.mean():.3f}, offsets {offset.tolist()} samples, "
        f"timing error max {numpy.abs(jitter[matched]).max()} samples, mean {numpy.abs(jitter[matched]).mean():.2f}"
    )
    print(f"  true hits per drum {numpy.bincount(drum).tolist()} (kick, snare, hat)")
    print(f"  melody SNR before {snr(tune, samples - tune):.1f} dB, after {snr(tune, residual - tune):.1f} dB")


def song(path):
    samples, rate = soundfile.read(path, dtype="float32", always_2d=True)
    report(samples.mean(axis=1), rate, "song")


def main():
    logger.start()
    if len(sys.argv) > 1:
        song(sys.argv[1])
    else:
        synthetic()


if __name__ == "__main__":
    main()
