# TODO

Decisions already made: Python (Flask, numpy, soundfile) server analyzes the
whole file up front and returns frames; Three.js page (CDN import map, no build)
plays the file locally via Web Audio and renders. Helix axis is vertical, one turn
per octave, so each pitch class is a vertical column; bars point radially outward,
length = magnitude. Analysis is testable on synthesized tones without HTTP.

## 2. Spectrogram endpoint

- Constant-Q spectrogram: 12 bins per octave, C1–C8 (96 bins), via librosa.cqt or a hand-rolled multi-resolution STFT so low notes still resolve to a semitone
- Frame rate ~60/s (hop chosen from the sample rate); magnitude → dB → normalize to 0..1 per file
- Return frames as binary float32 (frames × bins) with a small JSON header (sample_rate, hop, bins, fmin) instead of JSON arrays
- Tests: a synthesized 440 Hz tone lights the A4 bin and little else; a C-major chord lights three bins; silence is all zeros

## 3. Spiral renderer

- Open the page in a browser once: confirm Three.js loads from the CDN, the file plays, drag spins the ring (built by an agent with no browser)
- Fetch frames and decode the same file locally; start playback from an AudioBufferSourceNode
- Helix of 96 bars as one InstancedMesh: angle = 2π·(semitone/12), height = semitone/12 turns, bar length driven by the frame
- Per render: t = audioContext.currentTime − startTime + outputLatency; pick the frame at t, lerp between neighbours, decay bars so they fall smoothly
- Color bars by pitch class; label the 12 columns (C, C#, …) around the base
- Camera default looks at the helix from slightly above; OrbitControls gives click-and-drag to spin

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
