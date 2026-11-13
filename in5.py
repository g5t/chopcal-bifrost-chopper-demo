"""IN5 at the ILL, for `phasespace`: a steady source, six discs phased as ILL_IN5.instr does.

    import phasespace, in5
    phasespace.figure(in5.IN5, wavelength=4.5, speed=8500, ratio=0.5)

Everything here is McStas's ILL_IN5.instr (mcstas-comps/examples/ILL/ILL_IN5):

- the source is its Source_gen: three Maxwellians standing in for the reactor's cold source
  at the end of the shared H16 guide, which the instr does not model. It is steady, so the
  pulses are the choppers': every emission time is alike, and the time axis is time at the
  source, its zero arbitrary.
- the discs, two slits each, at the instr's distances, radii, slit widths and heights, in
  the guide's window at each; 1-2 and 5-6 are pairs 2 cm apart, 3 turns at `ratio` of the
  speed. Each disc's delay is the instr's: the time a neutron of `wavelength` takes from
  the source to it, so that one emitted at zero meets every slit centred.
- `Chopper0`, the instr's first disc at 0.2 m, is left out: it is not a chopper but the
  instr's time origin (`isfirst`), there to sample efficiently.

Two departures, both deliberate:

- `counter`, on by default, turns the second disc of each pair the other way. Every
  reference since IN5's upgrade around 2000 describes counter-rotating pairs; the instr
  turns all six the same way (DiskChopper's direction is the sign of `nu`, and all are
  positive). `counter=False` is the instr.
- the guide is perfect: IN5's own guide sections are not yet in (no `guide_table`).

The instr's `Ch_mean_R` is commented as rotation axis to guide axis, but DiskChopper takes
`radius` as the disc's and centres the beam at `radius - yheight/2`; this follows
DiskChopper, as McStas runs it.
"""
from __future__ import annotations

from math import ceil

import numpy as np
from chopcal import constants as c

from phasespace import Continuous, Instrument

#: Source_gen's Maxwellians: temperatures (K) and intensities, n/s per cm² per sr per Å
TEMPERATURES = (216.8, 33.9, 16.7)
INTENSITIES = (1.24e13, 1.02e13, 3.0423e12)
SOURCE_AREA = 14.0 * 22.0          # cm², its xwidth x yheight
#: Source_gen aims at a 38 x 200 mm target at 2.55 m (SourceTarget, its target_index)
SOLID_ANGLE = 0.038 * 0.2 / 2.55 ** 2


def maxwellian(wavelength, temperature):
    """Source_gen's SG_Maxwell: a Maxwellian flux spectrum normalised to 1 over λ, 1/Å."""
    a = 949.0 / temperature
    w = np.asarray(wavelength, dtype=float)
    return 2 * a * a * np.exp(-a / (w * w)) / w ** 5


def spectrum(wavelength):
    """What the source emits, n/s per Å per sr."""
    return SOURCE_AREA * sum(i * maxwellian(wavelength, t)
                             for t, i in zip(TEMPERATURES, INTENSITIES))


def _paths():
    """Each disc's distance from the source, m, as the instr's INITIALIZE adds them up:
    its Ch_Ltot, from Chopper0, plus Chopper0's 0.2 m."""
    gap, disc = 0.2130, 0.02                    # L_gap, disk_gap
    guide1, guide21, guide22, guide23 = 4.39, 0.695, 0.13, 0.695
    guide3, guide41, guide42, guide43, guide44 = 5.5125, 0.7425, 0.035, 0.75, 0.035
    at = [gap + guide1 + 0.0003 + guide21 + disc / 2]
    for step in (disc + guide22, disc + guide23 + guide3 + guide41 + 2 * 0.0003,
                 disc + guide42, disc + guide43, disc + guide44):
        at.append(at[-1] + step)
    return [0.2 + a for a in at]


NAMES = ('ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6')
PATH = dict(zip(NAMES, _paths()))
#: the sample, after Guide45, the monitor gap, the collimator and the sample chamber
SAMPLE = PATH['ch6'] + 0.01 + 0.79 + 0.03 + 0.13 + (0.24 - 0.025) + 0.025
SLITS = 2
#: Ch_alpha, the slit's angular width, degrees
SLIT = dict(zip(NAMES, (9.0, 9.0, 9.5, 9.5, 3.25, 3.25)))
#: Ch_mean_R and Ch_height, m: DiskChopper's radius and yheight
RADIUS = dict(zip(NAMES, (0.285, 0.285, 0.299, 0.299, 0.304, 0.304)))
HEIGHT = dict(zip(NAMES, (0.17, 0.16813, 0.081, 0.08031, 0.07069, 0.07)))
#: the guide's width and height where it meets each disc: the section before's exit
WINDOW = dict(zip(NAMES, ((0.03, 0.17), (0.03, 0.16813), (0.01579, 0.081),
                          (0.01568, 0.08031), (0.01411, 0.07069), (0.014, 0.07))))
#: the counter-rotating discs, when `counter`: the second of each pair
COUNTER = ('ch2', 'ch6')


def speeds(speed=8500, ratio=0.5, counter=True) -> dict:
    """Each disc's speed, Hz, signed, from the instr's `speed` (rpm) and `ratio`."""
    f = speed / 60
    out = {n: f for n in NAMES}
    out['ch3'] = f * ratio
    if counter:
        for n in COUNTER:
            out[n] = -f
    return out


def period(speed=8500, ratio=0.5) -> float:
    """How often a pulse reaches the sample, s: disc 3's slits, which pass every other
    one of the pairs' at ratio 1/2."""
    return 60 / (speed * min(ratio, 1.0) * SLITS)


class In5(Instrument):
    name = 'IN5'
    source = Continuous(spectrum)
    solid_angle = SOLID_ANGLE
    sample = SAMPLE
    names = NAMES
    groups = (0, 0, 1, 2, 3, 3)   # the two pairs share housings
    delay_ranges_ms = {n: (0.0, ceil(PATH[n] * 12 / c.H_OVER_M * 1e3)) for n in NAMES}
    guide_table = None
    wavelength_range = (2.0, 9.0)
    latest_emission = 2 * period()
    wavelength_bin = 0.01
    time_bin = 10e-6
    arrival_bin = 20e-6

    def choppers(self, wavelength=4.5, speed=8500, ratio=0.5, counter=True):
        """The instr's discs and phasing for `wavelength` (Å), `speed` (rpm) and `ratio`;
        `counter` turns the pairs against each other."""
        from chopcal.lib import Chopper, beam_aperture
        spin = speeds(speed, ratio, counter)
        velocity = c.H_OVER_M / wavelength
        out = []
        for n in self.names:
            half = SLIT[n] / 2
            edges = [-half, half, 180 - half, 180 + half]
            disc = Chopper(spin[n], PATH[n] / velocity, 0.0, edges, PATH[n],
                           beam_aperture(RADIUS[n], HEIGHT[n], *WINDOW[n]))
            out.append((disc, beam_aperture(RADIUS[n], HEIGHT[n], *WINDOW[n], at='centre')))
        return out

    def controls(self, wavelength=4.5, speed=8500, ratio=0.5, counter=True):
        from ipywidgets import Checkbox, FloatSlider, SelectionSlider
        ratios = [(f'1/{k}' if k > 1 else '1', 1 / k) for k in range(1, 7)]
        return {
            'wavelength': FloatSlider(value=wavelength, min=1.0, max=12.0, step=0.01,
                                      description='wavelength [Å]'),
            'speed': FloatSlider(value=speed, min=1000, max=17000, step=100,
                                 description='speed [rpm]'),
            'ratio': SelectionSlider(options=ratios, value=min(ratios, key=lambda r: abs(r[1] - ratio))[1],
                                     description='disc 3 ratio'),
            'counter': Checkbox(value=counter, description='counter-rotating pairs'),
        }

    def intended(self, polygons, settings, source):
        """The regions holding the chosen wavelength; the rest pass another one."""
        from chopcal.lib import wavelength_to_inverse_velocity
        u = wavelength_to_inverse_velocity(settings.get('wavelength', 4.5))
        return any(min(p[0] for p in q) <= u <= max(p[0] for p in q) for q in polygons)

    def zoom(self, **_):
        """Discs 3 to 6 and the sample."""
        return PATH['ch3'] - 0.4, SAMPLE + 0.3

    def note(self, result, values):
        spin = speeds(values['speed'], values['ratio'], values['counter'])
        turn = 'counter-rotating' if values['counter'] else 'co-rotating (the instr)'
        return (f"discs at {abs(spin['ch1']):.1f} Hz, disc 3 at {spin['ch3']:.1f} Hz, "
                f"pairs {turn}; a pulse every "
                f"{period(values['speed'], values['ratio']) * 1e3:.2f} ms")


IN5 = In5()


if __name__ == '__main__':
    from phasespace import transmission

    def inside(polygon, point):
        """Whether a convex polygon of (t, λ) holds a point, either winding."""
        p = np.asarray(polygon)
        d = np.roll(p, -1, axis=0) - p
        cross = d[:, 0] * (point[1] - p[:, 1]) - d[:, 1] * (point[0] - p[:, 0])
        return bool(np.all(cross >= -1e-12) or np.all(cross <= 1e-12))

    # the instr's numbers: discs 1 and 6 where INITIALIZE puts them, Source_gen's
    # Maxwellians normalised
    assert np.isclose(PATH['ch1'], 0.2 + 5.3083) and np.isclose(PATH['ch6'], 0.2 + 13.3089)
    w = np.linspace(0.05, 200.0, 400001)
    for t in TEMPERATURES:
        assert abs(np.trapezoid(maxwellian(w, t), w) - 1) < 1e-3, t
    # a steady source: the same rate whatever the window, spread evenly over it
    edges = np.arange(4.0, 5.0001, 0.01)
    one = IN5.source.intensity(edges, np.linspace(0, 1e-3, 11), SOLID_ANGLE)
    other = IN5.source.intensity(edges, np.linspace(0, 7e-3, 71), SOLID_ANGLE)
    assert np.allclose(one.sum(axis=0), other.sum(axis=0))
    assert np.allclose(other, other[0])
    assert np.isclose(one.sum(), spectrum(4.5) * SOLID_ANGLE, rtol=1e-3)

    result = transmission(IN5, dict(wavelength=4.5))
    assert result.transmitted > 0
    # the chosen neutron gets through once a period, in an intended region; half a
    # period on, the pairs are open again but disc 3, at half their speed, is not
    point = (period(), 4.5)
    assert any(main and inside(p, point) for p, main in zip(result.outline, result.main))
    assert not any(inside(p, (1.5 * period(), 4.5)) for p in result.outline)
    # the delays are the instr's: each disc's flight time at the wavelength
    for (disc, _), n in zip(IN5.choppers(wavelength=4.5), IN5.names):
        assert np.isclose(disc.delay, PATH[n] * 4.5 / c.H_OVER_M)
    # counter-rotation moves no opening, so the region is the same; the pairs' ramps
    # cancel instead of following each other, so less gets through
    co = transmission(IN5, dict(wavelength=4.5, counter=False))
    assert co.outline == result.outline
    assert result.transmitted < co.transmitted
    print(f'ok: {result.transmitted:.3g} n/s counter-rotating, '
          f'{co.transmitted:.3g} n/s co-rotating')
