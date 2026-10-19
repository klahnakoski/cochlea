import numpy

from cochlea.spectrogram import BINS, BINS_PER_OCTAVE, FRAME_RATE, FREQS

OFFSETS = numpy.round(12 * numpy.log2(numpy.arange(2, 11))).astype(int)  # 12 19 24 28 31 34 36 38 40
PRODUCT_OFFSETS = numpy.concatenate([[0], OFFSETS[:4]])  # partials 1..5 make a bin's salience
AUDIBLE = 0.1  # a fundamental within 20 dB of its overtone
CAP = 0.7  # an overtone louder than this times its fundamental keeps the excess
SALIENCE_FLOOR = 1e-3
# A fundamental in octave n (C1 = 1) claims only its first n overtones; bass has fewest.
FUNDAMENTALS = numpy.arange(BINS)[:, None] - OFFSETS  # (overtone bin, offset) -> fundamental bin
CLAIMABLE = (FUNDAMENTALS >= 0) & (numpy.arange(len(OFFSETS)) < FUNDAMENTALS // BINS_PER_OCTAVE + 1)
SECONDS_PER_WINDOW = 0.19  # 8192 points at 44.1 kHz, about 5.4 Hz per bin
STRIDE = 3  # STFT hop, in display frames
FMAX = FREQS[-1] * 2 ** (1 / 24)
HARMONIC = 2  # a fundamental's partials stand this far above the gaps between them
GATED = BINS_PER_OCTAVE  # C2: from here up the STFT resolves partials from gaps
PARTIALS = numpy.arange(1, len(PRODUCT_OFFSETS) + 1)
COMB = numpy.concatenate([PARTIALS, PARTIALS + 0.5])  # partials 1..5, then the gap above each


def suppress(linear, samples, sample_rate):
    """Hand each overtone's magnitude to its fundamental: (frames, bins) -> same shape.

    Harmonic product spectrum: a bin is an overtone when a sounding bin an overtone
    interval below has more salience (geometric mean over its first five partials),
    and, from C2 up, that bin's partials stand out of the STFT as a harmonic comb.
    """
    salience = numpy.exp(
        numpy.log(above(linear, PRODUCT_OFFSETS, floor=linear.max() * SALIENCE_FLOOR + 1e-30)).mean(axis=2)
    )
    harmonic = numpy.zeros_like(linear)
    harmonic[:, GATED:] = harmonicity(samples, sample_rate, frames=len(linear))
    pitched = (below(harmonic, OFFSETS) > HARMONIC) | (FUNDAMENTALS < GATED)
    fundamentals = below(linear, OFFSETS)
    rivals = below(salience, OFFSETS) * (fundamentals > AUDIBLE * linear[:, :, None]) * pitched * CLAIMABLE
    choice = rivals.argmax(axis=2)
    beaten = pick(rivals, choice) > salience
    moved = numpy.where(beaten, numpy.minimum(linear, CAP * pick(fundamentals, choice)), 0)
    return attribute(linear, moved=moved, offset=OFFSETS[choice])


def harmonicity(samples, sample_rate, frames):
    """(frames, bins from C2): geometric mean of partials 1..5 over the gaps above them, on the STFT."""
    spectra, resolution = stft(samples, sample_rate)
    index, weight = taps(FREQS[GATED:], COMB, bins=spectra.shape[1], resolution=resolution)
    levels = numpy.log(numpy.maximum((spectra[:, index] * weight).sum(axis=3), spectra.max() * SALIENCE_FLOOR + 1e-30))
    ratio = numpy.exp(levels[:, :, : len(PARTIALS)].mean(axis=2) - levels[:, :, len(PARTIALS) :].mean(axis=2))
    return ratio[numpy.minimum(numpy.round(numpy.arange(frames) / STRIDE).astype(int), len(ratio) - 1)]


def stft(samples, sample_rate):
    """(frames, bins) Hann-window magnitude below FMAX, centred every STRIDE display frames; and Hz per bin."""
    size = 1 << round(numpy.log2(SECONDS_PER_WINDOW * sample_rate))
    hop = round(sample_rate / FRAME_RATE) * STRIDE
    count = -(-len(samples) // hop)
    padded = numpy.pad(samples.astype("float32"), (size // 2, size))
    windows = numpy.lib.stride_tricks.sliding_window_view(padded, size)[::hop][:count]
    resolution = sample_rate / size
    top = int(FMAX / resolution) + 2
    spectrum = numpy.abs(numpy.fft.rfft(windows * numpy.hanning(size).astype("float32"), axis=1)[:, :top])
    return spectrum / (size / 4), resolution


def lobe(offset):
    """Hann window's magnitude response, in bins from its centre; 1 at 0, 0.5 at 1, 0 from 2."""
    near_one = numpy.isclose(numpy.abs(offset), 1)
    return numpy.where(near_one, 0.5, numpy.abs(numpy.sinc(offset) / numpy.where(near_one, 1, 1 - offset**2)))


def taps(pitches, harmonics, bins, resolution):
    """(index, weight), each (pitches, harmonics, 4): weights read a sinusoid's amplitude off its lobe."""
    position = pitches[:, None, None] * harmonics[:, None] / resolution
    index = numpy.floor(position).astype(int) + numpy.arange(-1, 3)
    shape = lobe(index - position)
    weight = shape / (shape**2).sum(axis=2, keepdims=True)
    inside = index[:, :, -1:] < bins
    return numpy.where(inside, index, 0), numpy.where(inside, weight, 0).astype("float32")


def below(grid, offsets):
    """(frames, bins, offsets): grid[t, b - offset], zero below the lowest bin."""
    index = numpy.arange(grid.shape[1])[:, None] - offsets
    return grid[:, index.clip(0)] * (index >= 0)


def above(grid, offsets, floor):
    """(frames, bins, offsets): grid[t, b + offset] at least `floor`, `floor` above the top bin."""
    index = numpy.arange(grid.shape[1])[:, None] + offsets
    return numpy.where(index < grid.shape[1], numpy.maximum(grid[:, index.clip(max=grid.shape[1] - 1)], floor), floor)


def pick(stack, choice):
    """stack[t, b, choice[t, b]]."""
    return numpy.take_along_axis(stack, choice[..., None], axis=2)[..., 0]


def attribute(linear, moved, offset):
    """Move `moved` magnitude (as power) from each bin to the root of its overtone chain."""
    frames, bins = linear.shape
    target = numpy.where(moved > 0, numpy.arange(bins) - offset, numpy.arange(bins))
    for _ in range(3):  # pointer jumping: chains up to 8 links
        target = numpy.take_along_axis(target, target, axis=1)
    power = moved**2
    cells = numpy.arange(frames)[:, None] * bins + target
    gained = numpy.bincount(cells.ravel(), weights=power.ravel(), minlength=frames * bins).reshape(frames, bins)
    return numpy.sqrt(numpy.maximum(linear**2 - power, 0) + gained).astype("float32")
