import numpy
from mo_logs import logger

from cochlea.loudness import gain_db

BINS_PER_OCTAVE = 12
BINS = 96
FMIN = 440.0 * 2 ** (-45 / 12)  # C1
FREQS = FMIN * 2 ** (numpy.arange(BINS) / BINS_PER_OCTAVE)
FRAME_RATE = 60
SEMITONE_REJECTION = 2.15  # 2π·Δf·τ at one semitone: neighbours sit 30 dB down
MAX_TAU = 0.04  # seconds; caps bass decay so beats stay sharp
TAUS = numpy.minimum(SEMITONE_REJECTION / (2 * numpy.pi * FREQS * (2 ** (1 / 12) - 1)), MAX_TAU)
BAND_EDGE = 4  # 2π·Δf·τ where the band is cut, about 49 dB down
HALF_WIDTHS = BAND_EDGE / (2 * numpy.pi * TAUS)  # Hz, centre to band edge
PAD_FRAMES = 120  # silence after the end, so circular wrap stays quiet
FLOOR_DB = 45
PHON = 80  # listening level for the equal-loudness contour
LOUDNESS = 10 ** (gain_db(FREQS, PHON) / 20)  # weight each bin as loud as the ear hears it


def spectrogram(samples, sample_rate):
    """Return (hop, frames): frames is a (frames, BINS) float32 array in 0..1."""
    hop, linear = magnitude(samples, sample_rate)
    return hop, normalize(linear)


def magnitude(samples, sample_rate):
    """Return (hop, linear): linear is a (frames, BINS) float32 magnitude array.

    One FFT of the whole file feeds every bin: each bin filters a band of that
    spectrum with a gammatone, of order 4, t³·e^(-t/τ)·e^(iωt), folds it to the frame
    grid, and an inverse FFT gives its envelope at every frame. The gammatone is
    causal, like the cochlea: a bar lights at its note's attack and decays after.
    """
    hop = round(sample_rate / FRAME_RATE)
    count = -(-len(samples) // hop)
    total = 1 << int(numpy.ceil(numpy.log2(count + PAD_FRAMES)))
    logger.info("spectrogram: {samples} samples, FFT size {size}", samples=len(samples), size=total * hop)
    spectrum = numpy.fft.rfft(samples.astype("float32"), n=total * hop)

    resolution = sample_rate / (total * hop)  # Hz per spectrum index
    fold = int(numpy.ceil(2 * HALF_WIDTHS.max() * hop / sample_rate))
    starts = numpy.round(FREQS / resolution).astype("int32") - fold * total // 2
    index = starts[:, None] + numpy.arange(fold * total, dtype="int32")
    detune = (2 * numpy.pi * (index * resolution - FREQS[:, None]) * TAUS[:, None]).astype("float32")
    pole = ((1 - 1j * detune) / (1 + detune * detune)).astype("complex64")  # (1 + i·detune)^-1
    pole[numpy.abs(detune) >= BAND_EDGE] = 0
    squared = pole * pole
    response = squared * squared  # order 4
    band = spectrum[index.clip(0, len(spectrum) - 1)] * response
    folded = band.reshape(BINS, fold, total).sum(axis=1)
    envelope = numpy.fft.ifft(folded, axis=1)[:, :count]
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
