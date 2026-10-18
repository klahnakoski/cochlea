# Cochlea

Audio visualizer. Plays a wav/flac/mp3 and draws its spectrum as a 3D bar chart
with 12 bars per octave, wrapped into a spiral so that one turn is one octave and
every pitch class lines up vertically — the way the inner ear lays frequency out
along its own spiral.

Analysis is a Fourier transform, followed by correlation across the spectrum to
suppress each instrument's overtones so the bars show fundamentals, not harmonics.
