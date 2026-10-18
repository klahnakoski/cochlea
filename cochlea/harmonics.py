import numpy

OFFSETS = numpy.round(12 * numpy.log2(numpy.arange(2, 11))).astype(int)  # 12 19 24 28 31 34 36 38 40
PRODUCT_OFFSETS = numpy.concatenate([[0], OFFSETS[:4]])  # partials 1..5 make a bin's salience
AUDIBLE = 0.1  # a fundamental within 20 dB of its overtone
CAP = 0.7  # an overtone louder than this times its fundamental keeps the excess
SALIENCE_FLOOR = 1e-3


def suppress(linear):
    """Hand each overtone's magnitude to its fundamental: (frames, bins) -> same shape.

    Harmonic product spectrum: a bin is an overtone when a sounding bin an overtone
    interval below has more salience (geometric mean over its first five partials).
    """
    salience = numpy.exp(
        numpy.log(above(linear, PRODUCT_OFFSETS, floor=linear.max() * SALIENCE_FLOOR + 1e-30)).mean(axis=2)
    )
    fundamentals = below(linear, OFFSETS)
    rivals = below(salience, OFFSETS) * (fundamentals > AUDIBLE * linear[:, :, None])
    choice = rivals.argmax(axis=2)
    beaten = pick(rivals, choice) > salience
    moved = numpy.where(beaten, numpy.minimum(linear, CAP * pick(fundamentals, choice)), 0)
    return attribute(linear, moved=moved, offset=OFFSETS[choice])


def below(grid, offsets):
    """(frames, bins, offsets): grid[t, b - offset], zero below the lowest bin."""
    index = numpy.arange(grid.shape[1])[:, None] - offsets
    return grid[:, index.clip(0)] * (index >= 0)


def above(grid, offsets, floor):
    """(frames, bins, offsets): grid[t, b + offset] at least `floor`, `floor` above the top bin."""
    index = numpy.arange(grid.shape[1])[:, None] + offsets
    return numpy.where(index < grid.shape[1], numpy.maximum(grid[:, index.clip(max=grid.shape[1] - 1)], floor), floor)


def pick(stack, choice):
    """stack[t, b, choice[t, b]]."""
    return numpy.take_along_axis(stack, choice[..., None], axis=2)[..., 0]


def attribute(linear, moved, offset):
    """Move `moved` magnitude (as power) from each bin to the root of its overtone chain."""
    frames, bins = linear.shape
    target = numpy.where(moved > 0, numpy.arange(bins) - offset, numpy.arange(bins))
    for _ in range(3):  # pointer jumping: chains up to 8 links
        target = numpy.take_along_axis(target, target, axis=1)
    power = moved**2
    cells = numpy.arange(frames)[:, None] * bins + target
    gained = numpy.bincount(cells.ravel(), weights=power.ravel(), minlength=frames * bins).reshape(frames, bins)
    return numpy.sqrt(numpy.maximum(linear**2 - power, 0) + gained).astype("float32")
