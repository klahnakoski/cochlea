# Cochlea

Audio visualizer. Plays a wav/flac/mp3 and draws its spectrum as a 3D bar chart
with 12 bars per octave, wrapped into a spiral so that one turn is one octave and
every pitch class lines up vertically — the way the inner ear lays frequency out
along its own spiral.

Analysis is a Fourier transform, followed by correlation across the spectrum to
suppress each instrument's overtones so the bars show fundamentals, not harmonics.

Needs Python 3.10+. Node is optional, for the page's parser test.

## Install

Mac:

```
git clone https://github.com/klahnakoski/cochlea.git
cd cochlea
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Windows (PowerShell):

```
git clone https://github.com/klahnakoski/cochlea.git
cd cochlea
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

## Run

With the venv active, on either platform:

```
python -m cochlea
```

Open http://localhost:5000 and choose an audio file.

## Test

With the venv active, from the repo root:

```
python -m pytest
node web/frames.test.mjs
```
