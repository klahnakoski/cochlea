import numpy
from mo_logs import logger

BINS_PER_OCTAVE = 12
BINS = 96
FMIN = 440.0 * 2 ** (-45 / 12)  # C1
FREQS = FMIN * 2 ** (numpy.arange(BINS) / BINS_PER_OCTAVE)
FRAME_RATE = 60
Q = 30  # resolution of a Q-cycle Hann window; neighbours stay dark
MIN_HALF_WIDTH = 10  # Hz; bass bands this wide so beats stay sharp
HALF_WIDTHS = numpy.maximum(2 * FREQS / Q, MIN_HALF_WIDTH)  # Hz, centre to band edge
PAD_FRAMES = 120  # silence after the end, so circular wrap stays quiet
FLOOR_DB = 60


def spectrogram(samples, sample_rate):
    """Return (hop, frames): frames is a (frames, BINS) float32 array in 0..1.

    One FFT of the whole file feeds every bin: each bin takes a Hann-shaped band
    of that spectrum around its frequency, folds it to the frame grid, and an
    inverse FFT gives its envelope at every frame centre.
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
    offset = ((index * resolution - FREQS[:, None]) / HALF_WIDTHS[:, None]).astype("float32")
    response = numpy.where(numpy.abs(offset) < 1, 0.5 + 0.5 * numpy.cos(numpy.pi * offset), 0)
    band = spectrum[index.clip(0, len(spectrum) - 1)] * response
    folded = band.reshape(BINS, fold, total).sum(axis=1)
    envelope = numpy.fft.ifft(folded, axis=1)[:, :count]
    logger.info("spectrogram: {frames} frames done", frames=count)
    return hop, normalize((numpy.abs(envelope) / (hop / 2)).T.astype("float32"))


def normalize(magnitude):
    """Map magnitude to dB below the file's peak, FLOOR_DB..0 -> 0..1."""
    peak = magnitude.max()
    if peak == 0:
        return numpy.zeros_like(magnitude)
    db = 20 * numpy.log10(numpy.maximum(magnitude, 1e-12) / peak)
    return ((db + FLOOR_DB) / FLOOR_DB).clip(0, 1).astype("float32")
