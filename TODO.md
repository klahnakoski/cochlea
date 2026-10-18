# TODO

- Set up the stack: Python server (soundfile decode, numpy analysis) serving a Three.js page; upload a file, get analysis back, play it in the browser
- Play any wav/flac/mp3 file and render in sync, with enough lookahead to absorb output latency
- Fourier transform into 12 bins per octave
- 3D bar chart wrapped in a spiral, one turn per octave, so each pitch class stacks vertically
- Correlation analysis to suppress each instrument's overtones so only fundamentals light up
- Later: click-and-drag to spin the spiral
