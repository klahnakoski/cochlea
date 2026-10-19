# TODO

Decisions already made: Python (Flask, numpy, soundfile) server analyzes the
whole file up front and returns frames; Three.js page (CDN import map, no build)
plays the file locally via Web Audio and renders. Helix axis is vertical, one turn
per octave, so each pitch class is a vertical column; bars point radially outward,
length = magnitude. Analysis is testable on synthesized tones without HTTP.

Wire format of POST /analyze (field `audio`), shared by stages 2 and 3: body is a
little-endian uint32 N, then N bytes of UTF-8 JSON header {sample_rate, hop,
frame_rate, fmin, bins_per_octave, bins, frames, duration, drums: {frame, level}}
(drums: the display frame and 0..1 loudness of each kick), then frames × bins
little-endian float32, row-major, values 0..1. Content-Type application/octet-stream.

## 4. Transport

- Play / pause / seek with a progress bar; file name and duration shown
- Drop a file anywhere on the page as well as the picker

## Crisper bars

- With suppression on, bars drop too fast to see: give every bar a decay from its previous value

## 5. Harmonic suppression

- Coherence (agent coherence): fold bars whose envelopes move in unison (modulation coherence on the STFT) to the lowest of the group
- Bass/kick split (agent bass-split): bass τ uncapped with a 3τ read-ahead is in cochlea/spectrogram.py. Still owed: rewrite test_bass_burst_stays_short and test_bar_lights_at_attack_and_fades_after (they encode the old short, causal bass), kick detection in cochlea/drums.py (low-band 30-160 Hz analytic-envelope transient: rise of 6 dB over 30 ms windows and a 3 dB fall 0.13-0.23 s later; prototype in the agent's scratchpad k5.py), a "drums" header field {frame, level}, the flash in web/main.js, and the tests
- After drums: peel instruments one at a time over the whole song. On the residual, group STFT peaks whose envelopes pulse together (modulation coherence in few-second windows, pooled over the song), take the strongest group as an instrument, subtract it, repeat
- Drum subtraction (cochlea/drums.py, scripts/drums.py: flux onsets, k-means on 0.3 s patches, aligned-average waveform templates, whole-song matched filter, banded least-squares gains) is weak: clusters are drum combinations (kick+hat, kick+snare+hat, hat), templates catch the next hit's tail, and averaging leaks the melody into them. Synthetic melody SNR -6 -> +5 dB, residual at hits -8.5 dB; song: 4 drums, 3261 hits, residual at hits -4 dB, only C1 octave mean drops (0.17 -> 0.04), 25 s for 317 s. Joint template least squares (block-Toeplitz CG) and re-averaging over all hits both made it worse (periodic hits make long templates ill-posed). Next: templates shorter than the hit spacing, or subtract on the magnitude grid; decide how to show drums and the residual
- Learn instruments from the whole song (scripts/instruments.py: harmonic-comb KL-NMF on an 8192-point STFT, fundamentals C3..B7 at 36/oct, Hann-lobe partials, 3 smooth unshifted percussion atoms, Wiener split of the display bars): not yet better than suppress. Learned curves still collapse to pure tones (song: one curve all-overtones 1.0, one 0.49, four pure), so overtones are explained as separate notes; synthetic false bars 0.4..1.0 vs suppress 0.06..0.3; song octave means barely change from plain; 39 s for the 300 s song (STFT matmuls 0.3 s each, BLAS dense beats numpy FFT). Next: stop curves collapsing (fewer or fixed curves, an overtone floor, or a pure-tone penalty), then cut iterations/bins for speed, before adding ?suppress=2
- Try the suppress checkbox on real recordings: CAP (cochlea/harmonics.py) is tuned on synthesized 1/h rolloff; instruments with a weak fundamental keep overtones above 0.7 of it
- Bass bins leak into the semitone above (bins 7.8 Hz apart, gammatone τ capped at 0.04 s): C3 lights C#3 at about 0.5

## 6. Polish

- Option for 36 bins per octave (3 per semitone) to show pitch drift
- Loudness floor and gain slider so quiet recordings fill the helix
- Package: pip install . gives a `cochlea` command that starts the server and opens the browser
