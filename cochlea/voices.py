"""Top notes: guess how many instruments play, follow that many notes, light only their keys.

Candidates are bars that top their neighbours, sit near the frame's loudest bar and,
from C3 up, show a harmonic comb on the STFT; below C3 the strongest bar counts when
it stands out; a quieter bar an overtone above one, struck with it, does not.
Instruments are counted as notes sounding together that started apart
(notes of one chord start together), a high percentile over the song. A voice holds
its key while the note sounds; an onset may move it to a nearby key; a free voice
prefers a note not struck with a note another voice holds.
"""

import numpy

from cochlea.harmonics import GATED, OFFSETS, harmonicity
from cochlea.spectrogram import BINS_PER_OCTAVE, FMIN, FRAME_RATE

BASS = 2 * BINS_PER_OCTAVE  # C3: below it the comb cannot resolve partials
LOUD = 0.5  # candidate floor, on the 0..1 display level
RANGE = 0.35  # a candidate is within this of the frame's loudest bar (about 16 dB)
STANDOUT = 0.1  # the strongest bass bar tops the bars two away by this
COMB = 1.5  # partials over gaps; looser than suppress, dense mixes blur the gaps
LASTING = 6  # frames a candidate must last to count an instrument
TOGETHER = 3  # frames apart that notes still start together
OVERTONE_LAG = 8  # frames an overtone's bar may start after its fundamental's
PERCENTILE = 80  # of instruments per frame, over frames with any
VOICES = (2, 8)
RISE = 0.12  # level gain over a bar's rise time that counts as an onset
FILTER_CYCLES = 5.75  # the gammatone τ is this many periods
RELEASE = 0.4  # a held key lets go below this level...
FADE = 0.45  # ...or this far below the frame's loudest bar
GAIN = 2.0  # times a candidate's level: worth of tracking it
FREE_JUMP = 1 / 120  # cost per semitone for a silent voice to take a note
JUMP = 1 / 12  # cost per semitone for an onset to move a sounding voice
STEAL = 1.0  # times the held note's level: cost to move a sounding voice
CHORD = 2.0  # cost per voice holding a note struck together with this one


def top(levels, samples, sample_rate):
    """Return (lit, voices): `levels` (frames, bars) with only the tracked keys kept, and the voice count."""
    keys, voices = follow(levels, samples, sample_rate)
    lit = numpy.zeros_like(levels)
    frame = numpy.broadcast_to(numpy.arange(len(levels))[:, None], keys.shape)
    sounding = keys >= 0
    lit[frame[sounding], keys[sounding]] = levels[frame[sounding], keys[sounding]]
    return lit, voices


def follow(levels, samples, sample_rate):
    """Return (keys, voices): keys is (frames, voices), the bar each voice holds, -1 when silent."""
    pitched = candidates(levels, samples, sample_rate)
    onset = onsets(levels)
    start, end = notes(levels >= RELEASE, onset=onset)
    candidate = pitched & ~overtones(levels, pitched=pitched, start=start)
    voices = instruments(candidate, start=start, end=end)
    return track(levels, candidate=candidate, start=start, onset=onset, voices=voices), voices


def candidates(levels, samples, sample_rate):
    """(frames, bars) bool: salient local maxima, comb-confirmed from C3, the standout bass bar below."""
    padded = numpy.pad(levels, ((0, 0), (2, 2)))
    peak = (levels >= padded[:, 1:-3]) & (levels >= padded[:, 3:-1])
    salient = peak & (levels >= LOUD) & (levels >= levels.max(axis=1, keepdims=True) - RANGE)
    comb = harmonicity(samples, sample_rate, frames=len(levels))[:, BASS - GATED :] > COMB
    bass = levels[:, :BASS]
    strongest = numpy.arange(BASS) == bass.argmax(axis=1)[:, None]
    standout = bass - numpy.maximum(padded[:, :BASS], padded[:, 4 : BASS + 4]) >= STANDOUT
    return salient & numpy.concatenate([strongest & standout, comb], axis=1)


def overtones(levels, pitched, start):
    """(frames, bars) bool: a pitched bar that is an overtone of a louder pitched bar below, struck together."""
    frame, bar = numpy.nonzero(pitched)
    frame, fundamental = frame[:, None], bar[:, None] - OFFSETS
    lower = fundamental.clip(0)
    lag = start[frame, bar[:, None]] - start[frame, lower]
    claimed = (fundamental >= 0) & pitched[frame, lower] & (levels[frame, lower] > levels[frame, bar[:, None]])
    overtone = numpy.zeros_like(pitched)
    overtone[frame[:, 0], bar] = (claimed & (lag >= -TOGETHER) & (lag <= OVERTONE_LAG)).any(axis=1)
    return overtone


def notes(sounding, onset):
    """(start, end), each (frames, bars): first and last frame of the note holding each sounding cell.

    A note runs while the bar sounds; an onset within it starts the next note.
    """
    frame = numpy.arange(len(sounding))[:, None]
    off = numpy.zeros_like(sounding[:1])
    begins = sounding & ~numpy.concatenate([off, sounding[:-1]]) | onset & ~numpy.concatenate([off, onset[:-1]])
    ends = sounding & ~numpy.concatenate([sounding[1:], off]) | numpy.concatenate([begins[1:], off])
    start = numpy.maximum.accumulate(numpy.where(begins, frame, -1), axis=0)
    end = numpy.minimum.accumulate(numpy.where(ends, frame, len(sounding))[::-1], axis=0)[::-1]
    return start, end


def instruments(candidate, start, end):
    """How many instruments: a high percentile of lasting notes per frame, notes started together counted once."""
    lasting = candidate & (end - start >= LASTING)
    never = len(candidate)
    starts = numpy.sort(numpy.where(lasting, start, never), axis=1)
    apart = (numpy.diff(starts, axis=1) > TOGETHER) & (starts[:, 1:] < never)
    count = lasting.any(axis=1) + apart.sum(axis=1)
    return int(numpy.clip(numpy.percentile(count[count > 0], PERCENTILE) if count.any() else 0, *VOICES))


def onsets(levels):
    """(frames, bars) bool: the level rose by RISE over the bar's rise time, as the Hits view does."""
    octave = numpy.arange(levels.shape[1]) / BINS_PER_OCTAVE
    rise = numpy.maximum(3, numpy.round(2 * FILTER_CYCLES * FRAME_RATE / (FMIN * 2**octave))).astype(int)
    frame = numpy.arange(len(levels))[:, None] - rise
    before = numpy.where(frame >= 0, levels[frame.clip(0), numpy.arange(levels.shape[1])], 0)
    return levels - before > RISE


def track(levels, candidate, start, onset, voices):
    """(frames, voices) bar each voice holds, -1 when silent.

    One pass over frames, numpy within each: tracking is sequential. A held key is
    kept while loud; free voices take open candidates, onsets may move held voices,
    greedily by cost, at most `voices` picks a frame. A frame whose candidates and
    onsets match the last, and where no key let go, keeps the last frame's keys.
    """
    floor = numpy.maximum(RELEASE, levels.max(axis=1) - FADE)
    contested = candidate & onset
    changed = numpy.concatenate([[True], (candidate[1:] != candidate[:-1]).any(axis=1) | contested[1:].any(axis=1)])
    key = numpy.full(voices, -1)
    struck = numpy.full(voices, -2 * TOGETHER - 1)  # start frame of each voice's note
    last = numpy.linspace(BINS_PER_OCTAVE, 6 * BINS_PER_OCTAVE, voices)  # spread silent voices over the range
    keys = numpy.full((len(levels), voices), -1)
    for t in range(len(levels)):
        row = levels[t]
        sounding = key >= 0
        held = sounding & (row[key] >= floor[t])
        if not changed[t] and (held == sounding).all():
            keys[t] = key
            continue
        key = numpy.where(held, key, -1)
        open_bar = candidate[t].copy()
        open_bar[key[held]] = False
        bars = numpy.flatnonzero(open_bar)
        if bars.size:
            begun = start[t, bars]
            distance = numpy.abs(bars - last[:, None])
            steal = numpy.where(onset[t, bars], STEAL * row[key][:, None] + JUMP * distance, numpy.inf)
            chord = CHORD * (held[:, None] & (numpy.abs(begun - struck[:, None]) <= TOGETHER)).sum(axis=0)
            cost = numpy.where(held[:, None], steal, FREE_JUMP * distance) + chord - GAIN * row[bars]
            for _ in range(min(voices, bars.size)):
                voice, choice = divmod(int(cost.argmin()), bars.size)
                if cost[voice, choice] >= 0:
                    break
                key[voice] = last[voice] = bars[choice]
                struck[voice] = begun[choice]
                cost[voice] = numpy.inf
                cost[:, choice] = numpy.inf
                cost[:, numpy.abs(begun - begun[choice]) <= TOGETHER] += CHORD
        keys[t] = key
    return keys
