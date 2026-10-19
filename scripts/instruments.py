"""Learn instruments (one overtone curve each, at any pitch) with harmonic-comb NMF on a short-time FFT.

A note at f0 predicts energy at k·f0 in the STFT, scaled by its instrument's curve.
Fundamentals start at C3; below, and under every bar, unshifted smooth percussion
atoms take the drums and noise. The fit only splits each display bar's energy:
percussion stays where it is, a note's share moves to its fundamental's bar.

Run from the repo root: python -m scripts.instruments [song.wav]
"""

import sys
import time

import numpy
import soundfile
from mo_logs import logger

from cochlea.harmonics import FMAX, STRIDE, lobe, stft, suppress
from cochlea.loudness import gain_db
from cochlea.spectrogram import BINS, BINS_PER_OCTAVE, FMIN, PHON, grid, magnitude, normalize
from scripts.harmonics import CASES as TONE_CASES, PARTIALS, RATE, score, signal

FIT = 36  # candidate pitches per octave
LOWEST = 2 * FIT  # C3: no fundamental below
HIGHEST = 7 * FIT  # C8: no fundamental at or above
PITCHES = grid(FIT)[0][LOWEST:HIGHEST]
HARMONICS = numpy.arange(1, 16)
INSTRUMENTS = 6
DRUMS = 3
BUMPS_PER_OCTAVE = 3  # percussion atoms are smooth at this resolution
LOUDEST_OVERTONE = 1  # times the fundamental
EPS = 1e-9


def partials(bins, resolution):
    """(index, weight), each (pitches, harmonics, 4): partial k of each pitch spread over the bins its lobe covers."""
    position = PITCHES[:, None, None] * HARMONICS[:, None] / resolution
    index = numpy.floor(position).astype(int) + numpy.arange(-1, 3)
    inside = index[:, :, -1:] < bins
    return numpy.where(inside, index, 0), numpy.where(inside, lobe(index - position), 0).astype("float32")


def gather(spectra, index, weight):
    """(frames, pitches, harmonics): spectra at each partial, interpolated."""
    return (spectra[:, index] * weight).sum(axis=3)


def comb(bins, index, weight):
    """(harmonics, pitches, bins): the partials as a dense tensor."""
    tensor = numpy.zeros((len(PITCHES), len(HARMONICS), bins), dtype="float32")
    pitch, harmonic, _ = numpy.indices(index.shape)
    numpy.add.at(tensor, (pitch, harmonic, index), weight)
    return tensor.transpose(1, 0, 2)


def bumps(freqs):
    """(bumps, bins): Gaussians in log frequency, BUMPS_PER_OCTAVE per octave from 20 Hz to FMAX."""
    octaves = numpy.log2(numpy.maximum(freqs, 10) / 20)
    centres = numpy.arange(0, numpy.log2(FMAX / 20) + 1e-9, 1 / BUMPS_PER_OCTAVE)
    return numpy.exp(-0.5 * ((octaves - centres[:, None]) * BUMPS_PER_OCTAVE) ** 2).astype("float32")


def fold(freqs):
    """(bins, BINS): each STFT bin split between the two display bars it lies between."""
    position = BINS_PER_OCTAVE * numpy.log2(numpy.maximum(freqs, 1) / FMIN)
    lower = numpy.floor(position).astype(int)
    upper_share = position - lower
    rows = numpy.arange(len(freqs))
    matrix = numpy.zeros((len(freqs) + 1, BINS + 1), dtype="float32")
    for at, share in ((lower, 1 - upper_share), (lower + 1, upper_share)):
        inside = (at >= 0) & (at < BINS)
        numpy.add.at(matrix, (numpy.where(inside, rows, -1), numpy.where(inside, at, -1)), share)
    return matrix[:-1, :-1]


def initial_curves():
    """Harmonic curves from bright (1/k^0.5) to mellow (1/k^2.5)."""
    return HARMONICS[None, :] ** -numpy.linspace(0.5, 2.5, INSTRUMENTS)[:, None].astype("float32")


def initial_drums(bumps):
    """(DRUMS, bumps): low, middle and high broad shapes."""
    centre = numpy.linspace(0, len(bumps), 2 * DRUMS + 1)[1::2]
    return numpy.exp(-0.5 * ((numpy.arange(len(bumps)) - centre[:, None]) / (len(bumps) / DRUMS)) ** 2)


class Model:
    """Harmonic-comb NMF of an STFT magnitude: notes (frames, instruments * pitches) and drums (frames, DRUMS)."""

    def __init__(self, spectra, resolution):
        freqs = numpy.arange(spectra.shape[1]) * resolution
        self.index, self.weight = partials(spectra.shape[1], resolution)
        self.comb = comb(spectra.shape[1], self.index, self.weight)
        self.bumps = bumps(freqs)
        self.fold = fold(freqs)
        self.curves = initial_curves()
        self.shapes = initial_drums(self.bumps).astype("float32")
        first = gather(spectra, self.index[:, :5], self.weight[:, :5])
        salience = numpy.exp(numpy.log(numpy.maximum(first, 1e-4 * spectra.max())).mean(axis=2))
        self.notes = numpy.tile(salience, INSTRUMENTS) / INSTRUMENTS
        self.drums = numpy.tile(spectra.mean(axis=1, keepdims=True), DRUMS) / DRUMS

    def atoms(self):
        """(instruments * pitches, bins)."""
        return numpy.tensordot(self.curves, self.comb, axes=1).reshape(-1, self.comb.shape[2])

    def percussion(self):
        """(DRUMS, bins), each peaking at 1."""
        shapes = self.shapes @ self.bumps
        return shapes / shapes.max(axis=1, keepdims=True)

    def fit(self, spectra, iterations, sparsity, learn):
        """KL multiplicative updates; `learn` also updates curves and drum shapes."""
        for _ in range(iterations):
            atoms, percussion = self.atoms(), self.percussion()
            ratio = spectra / (self.notes @ atoms + self.drums @ percussion + EPS)
            if learn:
                notes = self.notes.reshape(len(spectra), INSTRUMENTS, -1)
                numerator = numpy.einsum("tip,tpk->ik", notes, gather(ratio, self.index, self.weight), optimize=True)
                denominator = notes.sum(axis=0) @ self.weight.sum(axis=2) + EPS
                curves = self.curves * numerator / denominator
                self.curves = (curves / curves[:, :1]).clip(max=LOUDEST_OVERTONE)
                numerator = self.drums.T @ ratio @ self.bumps.T
                self.shapes = self.shapes * numerator / (self.drums.sum(axis=0)[:, None] * self.bumps.sum(axis=1) + EPS)
            self.notes = self.notes * (ratio @ atoms.T) / (atoms.sum(axis=1) + sparsity)
            self.drums = self.drums * (ratio @ percussion.T) / percussion.sum(axis=1)

    def bars(self, display):
        """(frames, BINS) display magnitude, each note's share moved to its fundamental's bar."""
        frames = len(display)
        at = numpy.minimum(numpy.arange(frames) / STRIDE, len(self.notes) - 1)
        lower = numpy.minimum(at.astype(int), len(self.notes) - 2).clip(0)
        upper_share = (at - lower)[:, None]
        notes = self.notes[lower] * (1 - upper_share) + self.notes[lower + 1] * upper_share
        drums = self.drums[lower] * (1 - upper_share) + self.drums[lower + 1] * upper_share
        note_bars = self.atoms() @ self.fold
        drum_bars = drums @ (self.percussion() @ self.fold) + EPS
        share = display**2 / (notes @ note_bars + drum_bars)
        power = (notes * (share @ note_bars.T)).reshape(frames, INSTRUMENTS, -1).sum(axis=1)
        moved = numpy.zeros((frames, BINS), dtype="float32")
        moved[:, LOWEST // 3 : HIGHEST // 3] = power.reshape(frames, -1, FIT // BINS_PER_OCTAVE).sum(axis=2)
        return numpy.sqrt(drum_bars * share + moved).astype("float32")


def learn(samples, sample_rate, sparsity=1.0, stride=4, curve_iterations=60, note_iterations=30):
    """A Model fit on the STFT; curves and drums learned on every `stride`th frame."""
    spectra, resolution = stft(samples, sample_rate)
    heard = 10 ** (gain_db(numpy.maximum(numpy.arange(spectra.shape[1]) * resolution, 20), PHON) / 20)
    spectra = (spectra * heard / (spectra * heard).max()).astype("float32")
    model = Model(spectra, resolution)
    notes, drums = model.notes, model.drums
    model.notes, model.drums = notes[::stride], drums[::stride]
    model.fit(spectra[::stride], curve_iterations, sparsity, learn=True)
    model.notes, model.drums = notes, drums
    model.fit(spectra, note_iterations, sparsity, learn=False)
    return model


def instruments(samples, sample_rate, display):
    """(frames, BINS) magnitude with overtones moved to their fundamentals."""
    return learn(samples, sample_rate).bars(display)


C3, C4, E4, G4, C5 = 36, 48, 52, 55, 60
BRIGHT = 1 / PARTIALS  # saw-like
MELLOW = numpy.array([1, 0.3, 0.1, 0, 0, 0, 0, 0, 0, 0])
KICK_BARS = slice(0, 24)  # below C3, where drums stay


def melody(bins, timbre, start, step, decay=0.8):
    return [
        dict(bin=b, start=start + step * i, stop=start + step * (i + 1), decay=decay, seconds=6.0, timbre=timbre)
        for i, b in enumerate(bins)
    ]


def kick(seconds=6.0, every=0.5):
    """Kick drums: a sine falling from 150 to 50 Hz, decaying in 0.25 s, plus a click."""
    t = numpy.arange(int(seconds * RATE)) / RATE
    age = (t - 0.2) % every
    phase = 2 * numpy.pi * (50 * age + 3 * (1 - numpy.exp(-age / 0.03)))
    click = numpy.random.default_rng(0).standard_normal(len(t)) * numpy.exp(-age / 0.005) * 0.3
    return ((numpy.sin(phase) + click) * numpy.exp(-age / 0.25) * (t > 0.2)).astype("float32")


CASES = {
    **{name: (notes, 0) for name, notes in TONE_CASES.items()},
    "bright melody": (melody([C4, E4, G4, C5, G4, E4], BRIGHT, start=0.2, step=0.9), 0),
    "mellow melody": (melody([C4, E4, G4, C5, G4, E4], MELLOW, start=0.2, step=0.9), 0),
    "bright bass + mellow melody": (
        melody([C3, G4 - 24, C3, G4 - 24, C3], BRIGHT, start=0.2, step=1.1)
        + melody([E4, G4, C5, G4, E4, C4], MELLOW, start=0.5, step=0.9),
        0,
    ),
    "mellow bass + bright melody": (
        melody([C3, G4 - 24, C3, G4 - 24, C3], MELLOW, start=0.2, step=1.1)
        + melody([E4, G4, C5, G4, E4, C4], BRIGHT, start=0.5, step=0.9),
        0,
    ),
    "octave: bright C3, mellow C4": (
        melody([C3] * 5, BRIGHT, start=0.2, step=1.1) + melody([C4, C4, E4, C4, C4], MELLOW, start=0.6, step=1.0),
        0,
    ),
    "octave: mellow C3, bright C4": (
        melody([C3] * 5, MELLOW, start=0.2, step=1.1) + melody([C4, C4, E4, C4, C4], BRIGHT, start=0.6, step=1.0),
        0,
    ),
    "kick + bright melody": (melody([C4, E4, G4, C5, G4, E4], BRIGHT, start=0.2, step=0.9), 2),
    "kick + mellow melody": (melody([C4, E4, G4, C5, G4, E4], MELLOW, start=0.2, step=0.9), 2),
}
METHODS = {
    "none": lambda samples, display: display,
    "suppress": lambda samples, display: suppress(display, samples=samples, sample_rate=RATE),
    "comb": lambda samples, display: instruments(samples, RATE, display),
}


def kicked(notes, loudness):
    """(samples, truth): `notes` with kicks `loudness` times the notes' peak under them."""
    samples, truth = signal(notes)
    if loudness:
        drums = kick(seconds=len(samples) / RATE)[: len(samples)]
        samples = samples + 0.3 * loudness * drums
        samples = (0.3 * samples / numpy.abs(samples).max()).astype("float32")
    return samples, truth


def compare():
    rows = []
    for name, (notes, loudness) in CASES.items():
        samples, truth = kicked(notes, loudness)
        _, display = magnitude(samples, RATE)
        for method, shape in METHODS.items():
            frames = normalize(shape(samples, display))
            recall, false, level = score(frames, truth)
            row = f"{name:30} {method:9} recall {recall:5.2f}  false bars {false:5.2f}  note level {level:4.2f}"
            if loudness:
                _, false_above, _ = score(frames[:, KICK_BARS.stop :], truth[:, KICK_BARS.stop :])
                row += f"  false above C3 {false_above:5.2f}  kick level {frames[:, KICK_BARS].max(axis=1).mean():4.2f}"
            rows.append(row)
    print("\n".join(rows))


def octave_levels(frames):
    return frames.reshape(len(frames), -1, BINS_PER_OCTAVE).mean(axis=(0, 2)).round(2).tolist()


def song(path):
    samples, rate = soundfile.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    start = time.monotonic()
    _, display = magnitude(mono, rate)
    spectrogram_seconds = time.monotonic() - start
    print("plain   ", octave_levels(normalize(display)))
    print("suppress", octave_levels(normalize(suppress(display, samples=mono, sample_rate=rate))))
    start = time.monotonic()
    model = learn(mono, rate)
    levels = octave_levels(normalize(model.bars(display)))
    print(f"comb     {levels}  spectrogram {spectrogram_seconds:.1f}s, comb {time.monotonic() - start:.1f}s")
    for i, curve in enumerate(model.curves):
        print(f"curve {i}: harmonics 2..5 " + " ".join(f"{level:.2f}" for level in curve[1:5]))
    for i, shape in enumerate(model.percussion() @ model.fold):
        print(f"drum {i}: per octave " + " ".join(f"{level:.2f}" for level in octave_levels(shape[None, :])))


def main():
    logger.start()
    if len(sys.argv) > 1:
        song(sys.argv[1])
    else:
        compare()


if __name__ == "__main__":
    main()
