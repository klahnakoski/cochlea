import numpy
from mo_logs import logger

BINS_PER_OCTAVE = 12
BINS = 96
FMIN = 440.0 * 2 ** (-45 / 12)  # C1
FRAME_RATE = 60
Q = 30  # window cycles per bin; wider than 17 so neighbours stay dark
FLOOR_DB = 60


def spectrogram(samples, sample_rate):
    """Return (hop, frames): frames is a (frames, BINS) float32 array in 0..1.

    Each bin is a Hann-windowed complex exponential Q cycles long, so every bin
    keeps semitone resolution; all are applied by FFT correlation.
    """
    hop = round(sample_rate / FRAME_RATE)
    count = -(-len(samples) // hop)
    freqs = FMIN * 2 ** (numpy.arange(BINS) / BINS_PER_OCTAVE)
    lengths = numpy.round(Q * sample_rate / freqs).astype(int)
    pad = lengths.max() // 2
    size = 1 << int(numpy.ceil(numpy.log2(len(samples) + 2 * pad + lengths.max())))
    padded = numpy.zeros(size, dtype="complex64")
    padded[pad : pad + len(samples)] = samples
    logger.info("spectrogram: {samples} samples, FFT size {size}", samples=len(samples), size=size)
    spectrum = numpy.fft.fft(padded)

    magnitude = numpy.zeros((count, BINS), dtype="float32")
    centres = numpy.arange(count) * hop + pad
    for k, (freq, length) in enumerate(zip(freqs, lengths)):
        n = numpy.arange(length)
        window = numpy.hanning(length) * numpy.exp(2j * numpy.pi * freq / sample_rate * n)
        correlation = numpy.fft.ifft(spectrum * numpy.conj(numpy.fft.fft(window, size)))
        magnitude[:, k] = numpy.abs(correlation[centres - length // 2]) / (length / 2)
        if (k + 1) % BINS_PER_OCTAVE == 0:
            logger.info(
                "spectrogram: octave {octave} of {octaves} done",
                octave=(k + 1) // BINS_PER_OCTAVE,
                octaves=BINS // BINS_PER_OCTAVE,
            )
    return hop, normalize(magnitude)


def normalize(magnitude):
    """Map magnitude to dB below the file's peak, FLOOR_DB..0 -> 0..1."""
    peak = magnitude.max()
    if peak == 0:
        return numpy.zeros_like(magnitude)
    db = 20 * numpy.log10(numpy.maximum(magnitude, 1e-12) / peak)
    return ((db + FLOOR_DB) / FLOOR_DB).clip(0, 1).astype("float32")
