# TODO

Decisions already made: Python (Flask, numpy, soundfile) server analyzes the
whole file up front and returns frames; Three.js page (CDN import map, no build)
plays the file locally via Web Audio and renders. Helix axis is vertical, one turn
per octave, so each pitch class is a vertical column; bars point radially outward,
length = magnitude. Analysis is testable on synthesized tones without HTTP.

Wire format of POST /analyze (field `audio`), shared by stages 2 and 3: body is a
little-endian uint32 N, then N bytes of UTF-8 JSON header {sample_rate, hop,
frame_rate, fmin, bins_per_octave, bins, frames, duration}, then frames × bins
little-endian float32, row-major, values 0..1. Content-Type application/octet-stream.

## 3. Spiral renderer

- Open the page in a browser once: confirm Three.js loads from the CDN, the file plays, the helix tracks the music, drag tilts the top of the axis; tune bar scale in web/main.js (built by an agent with no browser)

## 4. Transport

- Play / pause / seek with a progress bar; file name and duration shown
- Drop a file anywhere on the page as well as the picker

## Crisper bars

- Loudness weighting is +4 dB per octave (LOUDNESS_DB_PER_OCTAVE): judge on real music, maybe a real equal-loudness curve
- With suppression on, bars drop too fast to see: give every bar a decay from its previous value

## 5. Harmonic suppression

- Suppression over-claims on bass-heavy music: on a dance track it cuts mean level in octaves C3..C7 from 0.22 to 0.04, so voice vanishes; bass/kick energy is taken as the fundamental of everything above
- Try the suppress checkbox on real recordings: CAP (cochlea/harmonics.py) is tuned on synthesized 1/h rolloff; instruments with a weak fundamental keep overtones above 0.7 of it
- Bass bins leak into the semitone above (bins 7.8 Hz apart, gammatone τ capped at 0.04 s): C3 lights C#3 at about 0.5

## 6. Polish

- Option for 36 bins per octave (3 per semitone) to show pitch drift
- Loudness floor and gain slider so quiet recordings fill the helix
- Package: pip install . gives a `cochlea` command that starts the server and opens the browser
