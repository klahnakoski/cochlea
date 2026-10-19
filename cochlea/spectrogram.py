import numpy
from mo_logs import logger

from cochlea.loudness import gain_db

BINS_PER_OCTAVE = 12  # the display grid; finer grids split each bar evenly
OCTAVES = 8
FMIN = 440.0 * 2 ** (-45 / 12)  # C1
FRAME_RATE = 60
NEIGHBOUR_REJECTION = 2.15  # 2π·Δf·τ at the neighbouring bin: it sits 30 dB down
LEAD = 3  # τ each bin is read ahead: the gammatone's peak delay
TAIL = 12  # τ of silence after the end, so circular wrap stays quiet
BAND_EDGE = 4  # 2π·Δf·τ where the band is cut, about 49 dB down


def grid(bins_per_octave):
    """Return (freqs, taus) over C1..B8; a bar's sub-bins straddle its note."""
    split = bins_per_octave // BINS_PER_OCTAVE
    freqs = FMIN * 2 ** ((numpy.arange(OCTAVES * bins_per_octave) - (split - 1) / 2) / bins_per_octave)
    spacing = 2 ** (1 / bins_per_octave) - 1
    return freqs, NEIGHBOUR_REJECTION / (2 * numpy.pi * freqs * spacing)


FREQS, TAUS = grid(BINS_PER_OCTAVE)
BINS = len(FREQS)
FLOOR_DB = 45
PHON = 80  # listening level for the equal-loudness contour
FLAT_BELOW = FMIN * 4  # C3: below it, keep C3's weight so bass outweighs its own octave
LOUDNESS = 10 ** (gain_db(numpy.maximum(FREQS, FLAT_BELOW), PHON) / 20)  # weight each bin as loud as the ear hears it


def spectrogram(samples, sample_rate):
    """Return (hop, frames): frames is a (frames, BINS) float32 array in 0..1."""
    hop, linear = magnitude(samples, sample_rate)
    return hop, normalize(linear)


def emphasize(samples):
    """Tilt the music up 6 dB per octave (a first difference), so bass stops dominating."""
    return numpy.diff(samples, prepend=samples[:1]).astype("float32")


def magnitude(samples, sample_rate, bins_per_octave=BINS_PER_OCTAVE):
    """Return (hop, linear): linear is a (frames, bins) float32 magnitude array.

    One FFT of the whole file feeds every bin: each bin filters a band of that
    spectrum with a gammatone, of order 4, t³·e^(-t/τ)·e^(iωt), folds it to the frame
    grid, and an inverse FFT gives its envelope at every frame. Each envelope is
    read LEAD·τ ahead, so an attack peaks on time: bass bins (τ up to 0.18 s at C1)
    resolve semitones, and rise from the floor ahead of the attack and ring after.
    """
    freqs, taus = grid(bins_per_octave)
    return filterbank(samples, sample_rate=sample_rate, freqs=freqs, taus=taus)


def filterbank(samples, sample_rate, freqs, taus):
    """Return (hop, linear): one gammatone of time constant taus[i] at each of freqs, see magnitude."""
    half_widths = BAND_EDGE / (2 * numpy.pi * taus)  # Hz, centre to band edge
    hop = round(sample_rate / FRAME_RATE)
    count = -(-len(samples) // hop)
    lead = numpy.round(LEAD * taus * FRAME_RATE).astype("int32")
    total = 1 << int(numpy.ceil(numpy.log2(count + numpy.ceil(TAIL * taus.max() * FRAME_RATE))))
    logger.info("spectrogram: {samples} samples, FFT size {size}", samples=len(samples), size=total * hop)
    spectrum = numpy.fft.rfft(samples.astype("float32"), n=total * hop)

    resolution = sample_rate / (total * hop)  # Hz per spectrum index
    fold = int(numpy.ceil(2 * half_widths.max() * hop / sample_rate))
    starts = numpy.round(freqs / resolution).astype("int32") - fold * total // 2
    index = starts[:, None] + numpy.arange(fold * total, dtype="int32")
    detune = (2 * numpy.pi * (index * resolution - freqs[:, None]) * taus[:, None]).astype("float32")
    pole = ((1 - 1j * detune) / (1 + detune * detune)).astype("complex64")  # (1 + i·detune)^-1
    pole[numpy.abs(detune) >= BAND_EDGE] = 0
    squared = pole * pole
    response = squared * squared  # order 4
    band = spectrum[index.clip(0, len(spectrum) - 1)] * response
    folded = band.reshape(len(freqs), fold, total).sum(axis=1)
    envelope = numpy.fft.ifft(folded, axis=1)
    envelope = envelope[numpy.arange(len(freqs))[:, None], lead[:, None] + numpy.arange(count)]
    logger.info("spectrogram: {frames} frames done", frames=count)
    return hop, (numpy.abs(envelope) / (hop / 2)).T.astype("float32")


def normalize(magnitude):
    """Weight magnitude by LOUDNESS, then map dB below the file's peak, FLOOR_DB..0 -> 0..1."""
    magnitude = magnitude * LOUDNESS
    peak = magnitude.max()
    if peak == 0:
        return numpy.zeros_like(magnitude)
    db = 20 * numpy.log10(numpy.maximum(magnitude, 1e-12) / peak)
    return ((db + FLOOR_DB) / FLOOR_DB).clip(0, 1).astype("float32")


def bars(linear, bins_per_octave):
    """Fold a finer grid's magnitude to BINS bars: root of each bar's summed power."""
    split = bins_per_octave // BINS_PER_OCTAVE
    return numpy.sqrt((linear.reshape(len(linear), BINS, split) ** 2).sum(axis=2))
