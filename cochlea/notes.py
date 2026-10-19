"""Notes view: approximate the song's time-frequency image with horizontal rectangles, one key per rectangle.

The image is an STFT with a 1/4 s Gaussian window centred on each display frame,
sampled on the note axis (ROWS_PER_SEMITONE rows per semitone, C1 at row 0), in dB.
Beats are vertical stripes: pixels that rise above their time neighbourhood across
a broad band are unknown and take no part in the fit, so rectangles run through them.
Ridges (rows topping their neighbours) are pooled per key, and each key's contrast is
split into piecewise-constant runs by bottom-up merging with a fixed cost per run
(a Potts fit), so noise does not pay for its own run. Runs that stand out are
rectangles; one starting and ending with a lower one at a harmonic offset folds
into it. Each surviving rectangle lights its key.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy

from cochlea.harmonics import OFFSETS
from cochlea.spectrogram import BINS, FLOOR_DB, FMIN, FRAME_RATE

ROWS_PER_SEMITONE = 4
ROWS = BINS * ROWS_PER_SEMITONE
WINDOW_SECONDS = 0.25  # Gaussian with ±3σ inside it
IMAGE_FLOOR_DB = 60  # grey 0 is this far below the loudest pixel
IMAGE_RATE = 22050  # Hz the song is thinned to before the STFT
DECIMATION_TAPS = 63
CUTOFF = 0.9  # of the thinned Nyquist
CHUNK = 512  # frames per FFT batch
THREADS = 8
BEAT_REACH = 8  # frames each way: a stripe rises above the minimum within this
BEAT_RISE = 0.06  # mean grey above that minimum, about 4 dB
BEAT_SPAN = 3 * ROWS_PER_SEMITONE  # rows each way averaged: notes are thin, beats broad
BACKGROUND_SPAN = 6 * ROWS_PER_SEMITONE  # rows each way averaged as the local background
MARGIN = 0.06  # grey above the background before a pixel counts
RIDGE_SPAN = 2  # rows each way a ridge pixel must top
PENALTY = 0.1  # squared error a run must explain to pay for itself
LEVEL = 0.12  # mean contrast that makes a run a rectangle
TOGETHER = 4  # frames apart that an overtone may start or end


def rows_hz():
    """Frequency of each image row, C1 (row 0) .. B8, rows straddling each note."""
    return FMIN * 2 ** ((numpy.arange(ROWS) / ROWS_PER_SEMITONE - 0.5 + 0.5 / ROWS_PER_SEMITONE) / 12)


def decimated(samples, sample_rate):
    """(samples, sample_rate) low-passed and thinned to about IMAGE_RATE; B8 stays below the cutoff."""
    factor = max(1, sample_rate // IMAGE_RATE)
    if factor == 1:
        return samples.astype("float32"), sample_rate
    tap = numpy.arange(DECIMATION_TAPS) - DECIMATION_TAPS // 2
    cutoff = CUTOFF / factor
    low_pass = (numpy.sinc(tap * cutoff) * cutoff * numpy.hanning(DECIMATION_TAPS)).astype("float32")
    half = DECIMATION_TAPS // 2
    padded = numpy.pad(samples.astype("float32"), (half, half))
    piece = -(-len(samples) // (THREADS * factor)) * factor  # whole steps, so every piece keeps the phase

    def part(first):
        stretch = padded[first : min(first + piece, len(samples)) + 2 * half]
        return numpy.convolve(stretch, low_pass, mode="valid")[::factor]

    with ThreadPoolExecutor(THREADS) as pool:
        thinned = numpy.concatenate(list(pool.map(part, range(0, len(samples), piece))))
    return thinned, sample_rate / factor


def image(samples, sample_rate, frames):
    """(ROWS, frames) float32 grey 0..1: dB of the Gaussian STFT at each frame, IMAGE_FLOOR_DB below its peak."""
    samples, sample_rate = decimated(samples, sample_rate)
    hop = sample_rate / FRAME_RATE
    span = round(WINDOW_SECONDS * sample_rate)
    size = 1 << int(numpy.ceil(numpy.log2(span)))
    offsets = numpy.arange(span)
    window = numpy.exp(-0.5 * ((offsets - span / 2) / (span / 6)) ** 2).astype("float32")
    padded = numpy.concatenate([
        numpy.zeros(span // 2, "float32"),
        samples,
        numpy.zeros(span, "float32"),
    ])
    position = rows_hz() * size / sample_rate
    lower = numpy.floor(position).astype(int)
    upper_share = (position - lower).astype("float32")

    def batch(first):
        centres = numpy.round(numpy.arange(first, min(first + CHUNK, frames)) * hop).astype(int)
        spectrum = numpy.fft.rfft(padded[centres[:, None] + offsets] * window, n=size, axis=1)
        spectrum = numpy.abs(spectrum[:, : lower.max() + 2])
        return spectrum[:, lower] * (1 - upper_share) + spectrum[:, lower + 1] * upper_share

    with ThreadPoolExecutor(THREADS) as pool:
        magnitude = numpy.concatenate(list(pool.map(batch, range(0, frames, CHUNK)))).T
    db = 20 * numpy.log10(magnitude / max(magnitude.max(), 1e-30) + 1e-12)
    return ((db + IMAGE_FLOOR_DB) / IMAGE_FLOOR_DB).clip(0, 1).astype("float32")


def shifted(grey, by, axis):
    """`grey` moved `by` along axis, edges repeated."""
    count = grey.shape[axis]
    index = (numpy.arange(count) - by).clip(0, count - 1)
    return numpy.take(grey, index, axis=axis)


def box_mean(grey, reach):
    """Mean over rows within `reach` of each row."""
    total = numpy.concatenate([numpy.zeros_like(grey[:1]), numpy.cumsum(grey, axis=0, dtype="float64")])
    row = numpy.arange(len(grey))
    low, high = (row - reach).clip(0), (row + reach + 1).clip(max=len(grey))
    return ((total[high] - total[low]) / (high - low)[:, None]).astype("float32")


def running_min(grey, reach):
    """Minimum over frames within `reach` of each frame, by doubling windows."""
    size = 2 * reach + 1
    low = numpy.pad(grey, ((0, 0), (reach, reach)), constant_values=numpy.inf)
    width = 1
    while 2 * width <= size:
        low = numpy.minimum(low, shifted(low, -width, axis=1))
        width *= 2
    low = numpy.minimum(low, shifted(low, width - size, axis=1))  # low[t] = min over t .. t + size - 1
    return low[:, : grey.shape[1]]


def beats(grey):
    """(ROWS, frames) bool: unknown pixels, rising above their time neighbourhood along with a broad band."""
    band = box_mean(grey, BEAT_SPAN)
    return (band - running_min(band, BEAT_REACH) > BEAT_RISE) & (grey - running_min(grey, BEAT_REACH) > BEAT_RISE)


def ridges(grey):
    """(ROWS, frames) contrast over the local background, kept only where it tops the rows on both sides.

    The background is the mean over nearby rows at its quietest within BEAT_REACH frames, so beats do not raise it.
    """
    background = running_min(box_mean(grey, BACKGROUND_SPAN), BEAT_REACH)
    contrast = numpy.maximum(grey - background - MARGIN, 0)
    top = contrast > 0
    for by in range(1, RIDGE_SPAN + 1):
        top &= (contrast >= shifted(contrast, by, axis=0)) & (contrast >= shifted(contrast, -by, axis=0))
    top[:RIDGE_SPAN] = top[-RIDGE_SPAN:] = False
    return numpy.where(top, contrast, 0)


def fit(contrast, known):
    """Rectangles fitted to `contrast` on the known pixels, as a dict of arrays.

    Each row is cut into runs, every pixel its own run except runs of zeros and of
    unknowns; each pass merges every neighbouring pair whose merge raises the squared
    error least among its neighbours and by under PENALTY. Runs with a mean of at least
    LEVEL are rectangles: key (row), start and end (first and last known frame, or up to
    the neighbouring rectangle when unknown frames lie between them), weight
    (known pixels), total (summed contrast).
    """
    rows, frames = contrast.shape
    value = contrast.ravel().astype("float64")
    weight = known.ravel().astype("float64")
    zero = (value == 0) & (weight > 0)
    unknown = weight == 0
    starts = numpy.arange(value.size) % frames == 0
    starts[1:] |= ~((zero[1:] & zero[:-1]) | (unknown[1:] & unknown[:-1]))
    run = numpy.cumsum(starts) - 1
    first = numpy.flatnonzero(starts)
    sums = [numpy.bincount(run, weights=v) for v in (weight, weight * value, weight * value * value)]
    jitter = numpy.random.default_rng(0).random(first.size) * 1e-9  # breaks ties between equal costs
    while True:
        w, s, _ = sums
        mean = s / numpy.maximum(w, 1e-30)
        cost = w[:-1] * w[1:] / numpy.maximum(w[:-1] + w[1:], 1e-30) * (mean[:-1] - mean[1:]) ** 2
        cost = numpy.where(first[1:] % frames == 0, numpy.inf, cost + jitter[: cost.size])
        edge = numpy.full(1, numpy.inf)
        merge = (
            (cost < PENALTY)
            & (cost < numpy.concatenate([edge, cost[:-1]]))
            & (cost < numpy.concatenate([cost[1:], edge]))
        )
        if not merge.any():
            break
        keep = numpy.concatenate([[True], ~merge])
        run = numpy.cumsum(keep) - 1
        sums = [numpy.bincount(run, weights=v) for v in sums]
        first = first[keep]
    w, s, _ = sums
    end = numpy.concatenate([first[1:], [value.size]])
    rectangle = (w > 0) & (s >= LEVEL * w)
    known_pixel = numpy.flatnonzero(weight > 0)
    after_one = numpy.concatenate([[False], rectangle[:-1]]) & (first % frames != 0)
    before_one = numpy.concatenate([rectangle[1:], [False]]) & (end % frames != 0)
    start = numpy.where(
        after_one, first, known_pixel[numpy.searchsorted(known_pixel, first).clip(max=known_pixel.size - 1)]
    )[rectangle]
    last = numpy.where(before_one, end - 1, known_pixel[numpy.searchsorted(known_pixel, end) - 1])[rectangle]
    return {
        "key": start // frames,
        "start": start % frames,
        "end": last % frames,
        "weight": w[rectangle],
        "total": s[rectangle],
    }


def folded(rectangles, frames):
    """bool per rectangle: it starts and ends within TOGETHER frames of one an overtone offset below."""
    key, start, end = rectangles["key"], rectangles["start"], rectangles["end"]
    starting = numpy.full((BINS, frames), -1)
    starting[key, start] = numpy.arange(key.size)
    lag = numpy.arange(-TOGETHER, TOGETHER + 1)
    lower = (key[:, None, None] - OFFSETS[None, :, None]).repeat(lag.size, axis=2)
    when = (start[:, None, None] + lag).repeat(OFFSETS.size, axis=1)
    inside = (lower >= 0) & (when >= 0) & (when < frames)
    partner = numpy.where(inside, starting[lower.clip(0), when.clip(0, frames - 1)], -1)
    together = (partner >= 0) & (numpy.abs(end[partner] - end[:, None, None]) <= TOGETHER)
    return together.any(axis=(1, 2))


def loudness(rectangles, grey, known):
    """Mean known grey under each rectangle; `grey` and `known` are per key."""
    frames = grey.shape[1]
    flat = numpy.concatenate([[0], numpy.cumsum((grey * known).ravel(), dtype="float64")])
    first = rectangles["key"] * frames + rectangles["start"]
    after = rectangles["key"] * frames + rectangles["end"] + 1
    return (flat[after] - flat[first]) / rectangles["weight"]


def painted(rectangles, frames, level):
    """(BINS, frames): each rectangle filled with its `level`; rectangles on one key never overlap."""
    first = rectangles["key"] * frames + rectangles["start"]
    after = rectangles["key"] * frames + rectangles["end"] + 1
    step = numpy.bincount(first, weights=level, minlength=BINS * frames + 1)
    step -= numpy.bincount(after, weights=level, minlength=BINS * frames + 1)
    filled = numpy.cumsum(step)[:-1].reshape(BINS, frames)
    return numpy.where(filled > 1e-6, filled, 0)


def per_key(rows):
    """(BINS, ROWS_PER_SEMITONE, frames) view of a (ROWS, frames) image."""
    return rows.reshape(BINS, ROWS_PER_SEMITONE, rows.shape[1])


def stages(samples, sample_rate, frames):
    """Every step of the Notes view, by name: row images grey, known, contrast; key images; rectangles; keep."""
    grey = image(samples, sample_rate=sample_rate, frames=frames)
    known = ~beats(grey)
    contrast = ridges(grey)
    key_known = per_key(known).mean(axis=1) >= 0.5
    rectangles = fit(per_key(contrast).max(axis=1), key_known)
    return {
        "grey": grey,
        "known": known,
        "contrast": contrast,
        "key_grey": per_key(grey).max(axis=1),
        "key_known": key_known,
        "rectangles": rectangles,
        "keep": ~folded(rectangles, frames=frames),
    }


def notes(levels, samples, sample_rate):
    """Return (lit, header fields): `levels`-shaped display frames, each surviving rectangle lighting its key."""
    step = stages(samples, sample_rate=sample_rate, frames=len(levels))
    kept = {name: column[step["keep"]] for name, column in step["rectangles"].items()}
    grey = loudness(kept, grey=step["key_grey"], known=step["key_known"])
    display = ((grey * IMAGE_FLOOR_DB - IMAGE_FLOOR_DB + FLOOR_DB) / FLOOR_DB).clip(0, 1)
    lit = painted(kept, frames=len(levels), level=display).T.astype("float32")
    return lit, {"rectangles": len(display)}
