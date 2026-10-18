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

## Logging

- Server logs each analysis step: upload, decode, spectrogram progress, response

## 3. Spiral renderer

- Open the page in a browser once: confirm Three.js loads from the CDN, the file plays, the helix tracks the music, drag spins it; tune decay (0.02^dt) and bar scale in web/main.js (built by an agent with no browser)
- Analysis cost: 96 full-length FFTs per file — a multi-minute file takes tens of seconds and hundreds of MB; chunk or downsample per octave if it hurts

## 4. Transport

- Play / pause / seek with a progress bar; file name and duration shown
- Drop a file anywhere on the page as well as the picker

## 5. Harmonic suppression

- Prototype script (not the server) comparing methods on test tones and one real recording: (a) harmonic product spectrum baseline; (b) temporal correlation — a bin whose envelope co-varies with the bin 12, 19, 24, 28, 31 semitones below is an overtone, attribute its energy to the lower bin
- Pick the winner, implement as a server option (?suppress=1), with tests on a tone with synthesized overtones: fundamental stays, 2× and 3× bins drop
- Toggle in the UI

## 6. Polish

- Option for 36 bins per octave (3 per semitone) to show pitch drift
- Loudness floor and gain slider so quiet recordings fill the helix
- Package: pip install . gives a `cochlea` command that starts the server and opens the browser
