"""Find repeated drum samples and subtract them, leaving the pitched instruments.

A sampled drum sounds the same on every hit. Spectral-flux onsets on the display
grid give candidate hits; k-means on their time-frequency patches groups hits of
one drum; aligning and averaging a group's audio gives the drum's waveform, kept
only if the hits agree in phase; a matched filter over the whole song finds every
hit to the sample; least squares sets each hit's gain, overlapping hits solved
together, and subtracting the hits leaves the rest.
"""

import numpy
from numpy.lib.stride_tricks import sliding_window_view
from mo_logs import logger

from cochlea.spectrogram import FRAME_RATE, magnitude

KICK_BAND = (30, 160)  # Hz
KICK_WINDOW = 0.03  # seconds either side of a candidate hit
KICK_GAP = 0.05  # a hit tops the rise within this
KICK_RISE_DB = 6
KICK_FALL_DB = 3  # the envelope falls this much...
KICK_FALL_AFTER = (0.13, 0.23)  # ...this many seconds later
KICK_RANGE_DB = 24  # level 0 this far below the loudest hit
LEVEL_FLOOR_DB = 60  # below the file's peak, for flux and patches
FLUX_SPREAD = 3  # frames each side an onset must top
FLUX_BASELINE = 61  # frames in the moving median
FLUX_FACTOR = 1.5  # onset flux over the moving median
PATCH_FRAMES = 18  # 0.3 s at 60 frames per second
PATCH_LEAD = 1  # frames before the onset
CLUSTERS = 6
RESTARTS = 8
KMEANS_ROUNDS = 20
MIN_HITS = 8  # a cluster with fewer is not a repeated sample
MIN_SIMILARITY = 0.6  # mean correlation of a cluster's patches to its centroid
TRAINING_HITS = 64  # per drum, the patches closest to the centroid
PRE_SECONDS = 0.1  # audio before the onset frame, where the attack may be
SLACK_SECONDS = 0.025  # alignment search, each side
ALIGN_ROUNDS = 3
COHERENCE = 0.2  # template energy over mean hit energy: low means the hits differ in phase
TEMPLATE_SECONDS = 0.3
ATTACK = 0.1  # of the template envelope's peak
LEAD_SECONDS = 0.002  # template start before its attack
FADE_SECONDS = 0.01
DUPLICATE = 0.9  # normalized template correlation of one drum found twice
GAP_SECONDS = 0.04  # closest two hits of one drum
DETECT_FRACTION = 0.5  # of the median matched-filter score of a drum's surest hits
MIN_SCORE = 0.2  # normalized matched filter, below is never a hit
CG_ROUNDS = 200
PRUNE_ROUNDS = 3


class Kit:
    """Drum templates, one row per drum, and every hit, one entry per hit, sorted by time."""

    def __init__(self, templates, onset_hits, similarity, coherence, time, drum, score, gain):
        self.templates = templates  # (drums, samples)
        self.onset_hits = onset_hits  # onsets clustered to each drum
        self.similarity = similarity  # mean patch correlation to the drum's centroid
        self.coherence = coherence  # first template's energy over its hits' mean energy
        self.time = time  # sample where the template starts
        self.drum = drum
        self.score = score  # normalized matched filter
        self.gain = gain

    def render(self, length):
        """(drums, length): each drum's hits, template times gain, overlaps summed."""
        return render(self.templates, time=self.time, drum=self.drum, gain=self.gain, length=length)


def render(templates, time, drum, gain, length):
    """(drums, length): each drum's template at its hit times, scaled by gain, overlaps summed."""
    drums, width = templates.shape
    position = drum[:, None] * (length + width) + time[:, None] + numpy.arange(width)
    weights = gain[:, None] * templates[drum]
    total = numpy.bincount(position.ravel(), weights=weights.ravel(), minlength=drums * (length + width))
    return total.reshape(drums, length + width)[:, :length].astype("float32")


def separate(samples, rate):
    """Return the Kit of drums repeated in `samples`; subtract kit.render(...).sum(axis=0) to remove them."""
    samples = samples.astype("float32")
    hop, linear = magnitude(samples, rate)
    level = decibels(linear)
    frames = onsets(level)
    label, similarity = cluster(patches(level, frames))
    logger.info("drums: {hits} onsets", hits=len(frames))

    counts = numpy.bincount(label, minlength=CLUSTERS)
    means = numpy.bincount(label, weights=similarity, minlength=CLUSTERS) / numpy.maximum(counts, 1)
    kept = numpy.flatnonzero((counts >= MIN_HITS) & (means >= MIN_SIMILARITY))
    renumber = numpy.full(CLUSTERS, -1)
    renumber[kept] = numpy.arange(len(kept))
    member = renumber[label] >= 0
    templates, coherence = waveforms(
        samples, starts=frames[member] * hop, label=renumber[label[member]], similarity=similarity[member], rate=rate
    )
    counts, means = counts[kept], means[kept]
    distinct = numpy.flatnonzero((coherence >= COHERENCE) & ~duplicates(templates, counts))
    templates, counts, means, coherence = templates[distinct], counts[distinct], means[distinct], coherence[distinct]
    logger.info("drums: {drums} templates from clusters of {counts}", drums=len(templates), counts=counts.tolist())

    time, drum, score = detect(samples, templates=templates, counts=counts, rate=rate)
    windows = gather(samples, time, templates.shape[1])
    gain, keep = fit(windows, templates=templates, time=time, drum=drum)
    time, drum, score = time[keep], drum[keep], score[keep]
    logger.info("drums: {hits} hits", hits=len(time))
    return Kit(
        templates,
        onset_hits=counts,
        similarity=means,
        coherence=coherence,
        time=time,
        drum=drum,
        score=score,
        gain=gain,
    )


def decibels(linear):
    """Magnitude in dB, floored LEVEL_FLOOR_DB below the peak."""
    floor = linear.max() * 10 ** (-LEVEL_FLOOR_DB / 20) + 1e-20
    return 20 * numpy.log10(numpy.maximum(linear, floor))


def onsets(level):
    """Frames where the summed rise in level is a local peak, well over its moving median."""
    flux = numpy.concatenate([[0], numpy.maximum(numpy.diff(level, axis=0), 0).sum(axis=1)])
    spread = numpy.lib.stride_tricks.sliding_window_view(numpy.pad(flux, FLUX_SPREAD), 2 * FLUX_SPREAD + 1)
    half = FLUX_BASELINE // 2
    around = numpy.lib.stride_tricks.sliding_window_view(numpy.pad(flux, half, mode="edge"), FLUX_BASELINE)
    baseline = numpy.median(around, axis=1)
    peak = (flux == spread.max(axis=1)) & (flux > FLUX_FACTOR * baseline + 1e-6 * flux.max())
    return numpy.flatnonzero(peak)


def patches(level, frames):
    """(hits, PATCH_FRAMES * bins): each onset's level patch, zero mean, unit norm."""
    padded = numpy.pad(level, [(PATCH_LEAD, PATCH_FRAMES), (0, 0)], mode="edge")
    patch = padded[frames[:, None] + numpy.arange(PATCH_FRAMES)].reshape(len(frames), -1)
    patch = patch - patch.mean(axis=1, keepdims=True)
    return patch / numpy.maximum(numpy.linalg.norm(patch, axis=1, keepdims=True), 1e-12)


def cluster(patch):
    """(label, similarity) per hit: spherical k-means, RESTARTS run at once, the tightest kept."""
    rng = numpy.random.default_rng(0)
    count = min(CLUSTERS, len(patch))
    seeds = numpy.argsort(rng.random((RESTARTS, len(patch))), axis=1)[:, :count]
    centroids = numpy.pad(patch[seeds], [(0, 0), (0, CLUSTERS - count), (0, 0)])
    for _ in range(KMEANS_ROUNDS):
        similarity = numpy.einsum("rkd,nd->rnk", centroids, patch)
        label = similarity.argmax(axis=2)
        member = (label[:, :, None] == numpy.arange(CLUSTERS)).astype("float32")
        sums = numpy.einsum("rnk,nd->rkd", member, patch)
        norms = numpy.linalg.norm(sums, axis=2, keepdims=True)
        centroids = numpy.where(norms > 0, sums / numpy.maximum(norms, 1e-12), centroids)
    similarity = numpy.einsum("rkd,nd->rnk", centroids, patch)
    label = similarity.argmax(axis=2)
    best = similarity.max(axis=2).sum(axis=1).argmax()
    return label[best], similarity[best].max(axis=1)


def gather(samples, starts, length):
    """(hits, length): samples from each start, zero past the end."""
    padded = numpy.pad(samples, (0, length))
    return padded[starts[:, None] + numpy.arange(length)]


def waveforms(samples, starts, label, similarity, rate):
    """(templates, coherence) per label: its hits aligned by cross-correlation, averaged, trimmed to the attack."""
    drums = label.max() + 1
    order = numpy.lexsort([-similarity, label])
    first = numpy.searchsorted(label[order], numpy.arange(drums))
    rank = numpy.arange(len(order)) - first[label[order]]
    chosen = order[rank < TRAINING_HITS]
    medoid = starts[order[first]]  # most similar hit of each label
    starts, label = starts[chosen], label[chosen]

    pre, slack = round(PRE_SECONDS * rate), round(SLACK_SECONDS * rate)
    length = round(TEMPLATE_SECONDS * rate)
    span = pre + length
    padded = numpy.pad(samples, (pre + slack, span + slack))
    wide = padded[starts[:, None] + numpy.arange(span + 2 * slack)]
    mean = padded[medoid[:, None] + slack + numpy.arange(span)]
    member = (label == numpy.arange(drums)[:, None]).astype("float32")
    member /= member.sum(axis=1, keepdims=True)
    size = 1 << int(numpy.ceil(numpy.log2(2 * span + 2 * slack)))
    spectra = numpy.fft.rfft(wide, n=size)
    for _ in range(ALIGN_ROUNDS):
        correlation = numpy.fft.irfft(spectra * numpy.fft.rfft(mean, n=size)[label].conj(), n=size)
        offset = correlation[:, : 2 * slack + 1].argmax(axis=1)
        aligned = wide[numpy.arange(len(wide))[:, None], offset[:, None] + numpy.arange(span)]
        mean = member @ aligned
    coherence = (mean**2).sum(axis=1) / (member @ (aligned**2).sum(axis=1))

    width = round(0.002 * rate)
    power = numpy.pad(numpy.cumsum(mean**2, axis=1), [(0, 0), (width, 0)])
    envelope = power[:, width:] - power[:, :-width]
    attack = (envelope > ATTACK * envelope.max(axis=1, keepdims=True)).argmax(axis=1)
    start = numpy.clip(attack - round(LEAD_SECONDS * rate), 0, span - length)
    templates = mean[numpy.arange(drums)[:, None], start[:, None] + numpy.arange(length)]
    return (templates * taper(length, rate)).astype("float32"), coherence


def taper(length, rate):
    """Ones, with a half-cosine fade over the last FADE_SECONDS."""
    fade = round(FADE_SECONDS * rate)
    window = numpy.ones(length, dtype="float32")
    window[-fade:] = 0.5 + 0.5 * numpy.cos(numpy.pi * numpy.arange(fade) / fade)
    return window


def correlations(templates):
    """(drums, drums, size): R[a, b, l] = sum_k T_a[k] T_b[k + l], lag l modulo size."""
    size = 1 << int(numpy.ceil(numpy.log2(2 * templates.shape[1])))
    spectra = numpy.fft.rfft(templates, n=size)
    return numpy.fft.irfft(spectra.conj()[:, None, :] * spectra[None, :, :], n=size)


def duplicates(templates, counts):
    """Per drum: True where a more-hit drum has nearly the same waveform at some lag."""
    norms = numpy.linalg.norm(templates, axis=1)
    alike = correlations(templates).max(axis=2) / numpy.outer(norms, norms) > DUPLICATE
    index = numpy.arange(len(templates))
    better = (counts[None, :] > counts[:, None]) | (
        (counts[None, :] == counts[:, None]) & (index[None, :] < index[:, None])
    )
    return (alike & better).any(axis=1)


def detect(samples, templates, counts, rate):
    """(time, drum, score) of every hit: peaks of each drum's normalized matched filter, sorted by time."""
    drums, length = templates.shape
    size = 1 << int(numpy.ceil(numpy.log2(len(samples) + length)))
    spectrum = numpy.fft.rfft(samples, n=size)
    filtered = numpy.fft.irfft(spectrum * numpy.fft.rfft(templates, n=size).conj(), n=size)[:, : len(samples)]
    power = numpy.concatenate([[0], numpy.cumsum(samples.astype("float64") ** 2)])
    energy = power[numpy.minimum(numpy.arange(len(samples)) + length, len(samples))] - power[:-1]
    score = filtered / (numpy.linalg.norm(templates, axis=1)[:, None] * numpy.sqrt(numpy.maximum(energy, 1e-12)))
    logger.info("drums: matched filter of {drums} drums over {samples} samples", drums=drums, samples=len(samples))

    gap = round(GAP_SECONDS * rate)
    blocks = -(-len(samples) // gap)
    score = numpy.pad(score, [(0, 0), (0, blocks * gap - len(samples))]).reshape(drums, blocks, gap)
    where = score.argmax(axis=2)
    value = numpy.take_along_axis(score, where[:, :, None], axis=2)[:, :, 0]
    time = where + gap * numpy.arange(blocks)
    neighbour = numpy.pad(value, [(0, 0), (1, 1)], constant_values=-1)
    near = numpy.pad(time, [(0, 0), (1, 1)], constant_values=-(10**12))
    beaten = ((neighbour[:, :-2] > value) & (time - near[:, :-2] < gap)) | (
        (neighbour[:, 2:] >= value) & (near[:, 2:] - time < gap)
    )
    surest = -numpy.sort(-value, axis=1)[numpy.arange(drums), numpy.minimum(counts, blocks) // 2]
    threshold = numpy.maximum(DETECT_FRACTION * surest, MIN_SCORE)
    drum, block = numpy.nonzero(~beaten & (value > threshold[:, None]))
    order = numpy.argsort(time[drum, block], kind="stable")
    return time[drum, block][order], drum[order], value[drum, block][order]


def neighbours(time, length):
    """(near, valid), each (hits, band): hits starting less than `length` from each hit; invalid slots point home."""
    low = numpy.searchsorted(time, time - length + 1)
    high = numpy.searchsorted(time, time + length)
    near = low[:, None] + numpy.arange((high - low).max())
    valid = near < high[:, None]
    return numpy.where(valid, near, numpy.arange(len(time))[:, None]), valid


def fit(windows, templates, time, drum):
    """(gain, keep): least-squares gain of each hit, overlaps solved together; keep drops non-positive gains."""
    keep = numpy.arange(len(time))
    gain = numpy.zeros(len(time))
    lag = correlations(templates)
    for _ in range(PRUNE_ROUNDS):
        gain[:] = 0
        gain[keep] = solve(windows[keep], templates=templates, lag=lag, time=time[keep], drum=drum[keep])
        keep = keep[gain[keep] > 0]
    return gain[keep], keep


def solve(windows, templates, lag, time, drum):
    """Gains g minimizing |samples - sum_i g_i T_drum_i(. - time_i)|^2, by conjugate gradient on the banded Gram."""
    target = (windows * templates[drum]).sum(axis=1).astype("float64")
    near, valid = neighbours(time, templates.shape[1])
    gram = numpy.where(valid, lag[drum[:, None], drum[near], (time[:, None] - time[near]) % lag.shape[2]], 0)
    return conjugate_gradient(lambda g: (gram * g[near]).sum(axis=1), target)


def conjugate_gradient(operator, target):
    """Solve operator(x) = target for a symmetric positive operator."""
    x = numpy.zeros_like(target)
    residual = target - operator(x)
    direction = residual.copy()
    size = residual @ residual
    goal = 1e-12 * (target @ target)
    for _ in range(CG_ROUNDS):
        if size <= goal:
            break
        step = operator(direction)
        alpha = size / max(direction @ step, 1e-300)
        x += alpha * direction
        residual -= alpha * step
        new_size = residual @ residual
        direction = residual + (new_size / size) * direction
        size = new_size
    return x


def kicks(samples, sample_rate):
    """Return (frames, levels): the display frame and 0..1 loudness of each kick, in time order."""
    hop = round(sample_rate / FRAME_RATE)
    divisors = numpy.arange(1, hop + 1)
    factor = divisors[(hop % divisors == 0) & (hop // divisors >= 32)].max()
    small = numpy.pad(samples, (0, -len(samples) % factor)).reshape(-1, factor).mean(axis=1)
    rate = sample_rate / factor
    spectrum = numpy.fft.fft(small)
    freq = numpy.fft.fftfreq(len(small), 1 / rate)
    spectrum[(freq < KICK_BAND[0]) | (freq > KICK_BAND[1])] = 0
    power = numpy.abs(numpy.fft.ifft(spectrum)) ** 2

    width = round(KICK_WINDOW * rate)
    now, before = window_db(power, 0, width), window_db(power, -width, width)
    rise = now - before
    gap = round(KICK_GAP * rate)
    around = sliding_window_view(numpy.pad(rise, gap, constant_values=-numpy.inf), 2 * gap + 1)
    peak = (rise >= around[:, : gap + 1].max(axis=1)) & (rise > around[:, gap + 1 :].max(axis=1))  # last of a tie
    top = sliding_window_view(numpy.pad(now, (0, gap), mode="edge"), gap)[: len(now)].max(axis=1)
    late, later = (round(a * rate) for a in KICK_FALL_AFTER)
    fall = top - window_db(power, late, later - late)
    at = numpy.flatnonzero(peak & (rise >= KICK_RISE_DB) & (fall >= KICK_FALL_DB))
    frames, first = numpy.unique((at + width // 2) * factor // hop, return_index=True)
    loud = top[at[first]]
    return frames, (1 + (loud - loud.max(initial=-numpy.inf)) / KICK_RANGE_DB).clip(0, 1)


def window_db(power, start, length):
    """Return the dB of mean power over [t+start, t+start+length) for every t."""
    total = numpy.concatenate([[0], numpy.cumsum(power)])
    t = numpy.arange(len(power)) + start
    mean = (total[(t + length).clip(0, len(power))] - total[t.clip(0, len(power))]) / length
    return 10 * numpy.log10(mean + 1e-9 * power.max() + 1e-30)
