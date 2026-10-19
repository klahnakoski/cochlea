import numpy

from cochlea.drums import separate

RATE = 44100


def loop():
    """(samples, starts): an identical kick every 0.5 s and a hat between, 12 s."""
    t = numpy.arange(round(0.3 * RATE)) / RATE
    kick = numpy.sin(2 * numpy.pi * (50 * t + 3 * (1 - numpy.exp(-t / 0.03)))) * numpy.exp(-t / 0.1)
    hat = 0.2 * numpy.diff(numpy.random.default_rng(2).standard_normal(len(t) + 1)) * numpy.exp(-t / 0.02)
    starts = numpy.arange(24) * RATE // 2 + RATE // 4
    samples = numpy.zeros(13 * RATE)
    samples[starts[:, None] + numpy.arange(len(t))] += kick
    samples[starts[:, None] + RATE // 4 + numpy.arange(len(t))] += hat
    return samples.astype("float32"), numpy.concatenate([starts, starts + RATE // 4])


def test_loop_hits_found_and_subtracted():
    samples, starts = loop()
    kit = separate(samples, RATE)
    residual = samples - kit.render(len(samples)).sum(axis=0)
    assert len(kit.templates) >= 2
    found = numpy.abs(kit.time[:, None] - starts).min(axis=0) < 0.005 * RATE
    assert found.mean() > 0.9
    assert 10 * numpy.log10((residual**2).sum() / (samples**2).sum()) < -15
