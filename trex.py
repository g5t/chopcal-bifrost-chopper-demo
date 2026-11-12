"""T-REX, for `phasespace`: beamline W7, six discs phased as trex_primary.instr does.

    import phasespace, trex
    phasespace.figure(trex.TREX, wavelength=2.4, rrm=4, mode='HF')

The phasing is the instr's INITIALIZE: every disc's delay is the time a neutron of the
central wavelength L0, emitted `T_OFFSET` into the pulse, takes to reach the disc's
phasing distance. The bandwidth pair turns at 14 Hz, the pulse-shaping pair against each
other at 3/4 of RRM x 14 Hz, the monochromating pair at RRM x 14 Hz; HF opens the P and M
discs' wider slits, HR their narrower ones. The P pair and M1 hang above the beam, the
rest below it. Distances, slits and windows are the instr's;
the phasing distances are its own numbers, the discs' paths where it places them.
"""
from __future__ import annotations

from chopcal import constants as c

import ess_butterfly
from phasespace import Instrument

#: Into the pulse that the discs are phased for, s: about its centre.
T_OFFSET = 1.7e-3
FREQUENCY = 14.0
#: Slit edges, degrees from each disc's zero mark, for HR; HF adds `HF_SHIFT`.
EDGES = {'bw1': [-30.7, 30.7], 'bw2': [-31.65, 31.65],
         'p1': [-10, 10, 107.5, 142.5, 170, 190, 287.5, 322.5],
         'p2': [-10, 10, 107.5, 142.5, 170, 190, 287.5, 322.5],
         'm1': [178.25, 181.25, 362.8, 367.2], 'm2': [-1.25, 1.25, 182.8, 187.2]}
HF_SHIFT = {'p1': 55, 'p2': 55, 'm1': 175, 'm2': 175}
#: Where the beam crosses each disc, degrees from its zero mark: 180 hangs the disc above
#: the beam. M1 and M2 turn the same way on opposite sides of it, so their edges cross the
#: beam in opposite directions -- to the beam they are a counter-rotating pair.
BEAM = {'bw1': 0, 'bw2': 0, 'p1': 180, 'p2': 180, 'm1': 180, 'm2': 0}
#: What the delays are calculated for, m from the moderator.
PHASING = {'bw1': 32.0, 'bw2': 40.0, 'p1': 107.95, 'p2': 108.05,
           'm1': 162.0325 - 0.005, 'm2': 162.0325 + 0.005}
#: Where the instr puts the discs, m from the moderator along the guide.
PATH = {'bw1': 31.96404, 'bw2': 39.95505, 'p1': 107.95, 'p2': 108.05,
        'm1': 162.0275, 'm2': 162.0375}
#: Each disc's slit height, then the width and height of the guide at it, m.
WINDOW = {'bw1': (0.085, 0.06, 0.083061), 'bw2': (0.085, 0.06, 0.084983),
          'p1': (0.095, 0.06, 0.085), 'p2': (0.095, 0.06, 0.085),
          'm1': (0.045, 0.022433, 0.034987), 'm2': (0.045, 0.022433, 0.034987)}
RADIUS = 0.35


#: Repetition rate multiplications: multiples of 4, so that every disc, the P pair at
#: 3/4 of RRM x 14 Hz among them, turns at a multiple of 14 Hz.
RRM = tuple(range(4, 25, 4))


def speeds(rrm: int) -> dict:
    """Each disc's speed, Hz, signed."""
    p, m = 0.75 * rrm * FREQUENCY, rrm * FREQUENCY
    return {'bw1': FREQUENCY, 'bw2': FREQUENCY, 'p1': p, 'p2': -p, 'm1': m, 'm2': m}


def repetition(rrm: int) -> float:
    """How far apart in wavelength the repetitions are, Å: as the instr has it, for the
    monochromating discs' distance."""
    return c.H_OVER_M / (FREQUENCY * rrm * 162)


class Trex(Instrument):
    name = 'T-REX'
    beamline = ess_butterfly.Beamline('W', 7)
    #: the first guide's entrance, 89.955 x 29.992 mm at 1.89593 m from the moderator
    solid_angle = 0.089955 * 0.029992 / 1.89593 ** 2
    sample = 163.8
    names = tuple(EDGES)
    groups = (0, 1, 2, 2, 3, 3)
    #: T_OFFSET plus the flight time at up to 8 Å
    delay_ranges_ms = {n: (0.0, round(1e3 * T_OFFSET + PHASING[n] * 8 / c.H_OVER_M * 1e3 + 5))
                       for n in EDGES}
    guide_table = None   # ponytail: perfect guide until T-REX has a McStas scan of its own
    wavelength_bin = 0.005
    time_bin = 10e-6
    arrival_bin = 10e-6   # a band takes a few hundred µs to arrive
    wavelength_range = (1.0, 4.0)
    latest_emission = 6e-3

    def choppers(self, wavelength=2.4, rrm=4, mode='HF', flip_m2=False):
        """The instr's phasing for central wavelength `wavelength` (Å), repetition rate
        multiplication `rrm`, and 'HF' or 'HR'.

        In the instr both of M1's openings line up with M2's in either mode -- as though
        one disc were mounted the other way round from the design. `flip_m2` turns M2
        over: its openings mirror, a -> 360 - a, still phased to put the selected slit
        on the beam, so only the selected pair lines up and the other pair is 10 degrees
        apart.
        """
        from chopcal.lib import Chopper, beam_aperture
        if mode not in ('HF', 'HR'):
            raise ValueError(f"mode is 'HF' or 'HR', not {mode!r}")
        if rrm not in RRM:
            raise ValueError(f'rrm is one of {RRM}, not {rrm!r}')
        speed = speeds(rrm)
        shift = HF_SHIFT if mode == 'HF' else {}
        velocity = c.H_OVER_M / wavelength
        out = []
        for n in self.names:
            inner = beam_aperture(RADIUS, *WINDOW[n])
            centre = beam_aperture(RADIUS, *WINDOW[n], at='centre')
            edges = [e + shift.get(n, 0) for e in EDGES[n]]
            if flip_m2 and n == 'm2':
                edges = sorted(360 - e for e in edges)
            spin, beam = speed[n], BEAM[n]
            if beam == 180:
                # chopcal takes every disc as below the beam, which sets which way its
                # edges cross it: the same disc seen from below turns the other way, with
                # its angles mirrored -- every edge still crosses the beam when it did
                spin, beam, edges = -spin, -beam, sorted(-e for e in edges)
            disc = Chopper(spin, T_OFFSET + PHASING[n] / velocity, beam, edges, PATH[n],
                           inner)
            out.append((disc, centre))
        return out

    def controls(self, wavelength=2.4, rrm=4, mode='HF', flip_m2=False):
        from ipywidgets import Checkbox, FloatSlider, SelectionSlider, ToggleButtons
        return {
            'wavelength': FloatSlider(value=wavelength, min=0.5, max=8.0, step=0.01,
                                      description='central λ, L0 [Å]'),
            'rrm': SelectionSlider(options=RRM, value=rrm, description='RRM'),
            'mode': ToggleButtons(options=['HF', 'HR'], value=mode, description='mode'),
            'flip_m2': Checkbox(value=flip_m2, description='M2 turned over'),
        }

    def zoom(self, wavelength=2.4, rrm=4, mode='HF', flip_m2=False):
        """The M pair and the sample, around the L0 band: from a band spacing before it
        reaches 157 m to one after it reaches 164.5 m."""
        low, high = 157.0, 164.5
        spacing = 1 / (rrm * FREQUENCY)
        arrive = [T_OFFSET + at * wavelength / c.H_OVER_M for at in (low, high)]
        return low, high, arrive[0] - spacing, arrive[1] + spacing

    def note(self, result, values):
        speed = speeds(values['rrm'])
        return (f"P at {abs(speed['p1']):g} Hz, M at {speed['m1']:g} Hz; "
                f"repetitions {repetition(values['rrm']):.3f} Å apart")


TREX = Trex()


if __name__ == '__main__':
    import numpy as np
    from phasespace import transmission

    def inside(polygon, point):
        """Whether a convex polygon of (t, λ) holds a point, either winding."""
        p = np.asarray(polygon)
        d = np.roll(p, -1, axis=0) - p
        cross = d[:, 0] * (point[1] - p[:, 1]) - d[:, 1] * (point[0] - p[:, 0])
        return bool(np.all(cross >= -1e-12) or np.all(cross <= 1e-12))

    result = transmission(TREX, dict(wavelength=2.4, rrm=4, mode='HF'))
    assert result.transmitted > 0
    # the central neutron, L0 emitted at T_OFFSET, gets through, and so do the
    # repetitions either side of it, all in intended regions
    for k in (-1, 0, 1):
        point = (T_OFFSET, 2.4 + k * repetition(4))
        assert any(main and inside(p, point)
                   for p, main in zip(result.outline, result.main)), k
    # HR lets less through than HF; parking the M pair, more
    assert transmission(TREX, dict(mode='HR')).transmitted < result.transmitted
    assert transmission(TREX, parked=('m1', 'm2')).transmitted > result.transmitted
    # the delays are the instr's: T_OFFSET plus the central neutron's flight
    for (disc, _), n in zip(TREX.choppers(wavelength=2.4), TREX.names):
        assert np.isclose(disc.delay, T_OFFSET + PHASING[n] * 2.4 / c.H_OVER_M)
    # the M zoom holds the central neutron's arrival at the M pair, and no more than the
    # neighbouring bands; the arrival grid resolves a band: many bins with a σλ
    low, high, start, end = result.zoom
    at_m = T_OFFSET + PATH['m1'] * 2.4 / c.H_OVER_M
    assert low < PATH['m1'] < high and start < at_m < end
    assert end - start < 3 / (4 * FREQUENCY) + (high - low) * 2.4 / c.H_OVER_M
    mean, sigma = result.spread
    assert np.sum(np.isfinite(sigma) & (abs(mean - 2.4) < 0.1)) > 10
    # turning M2 over keeps the selected pair of M openings lined up, so it changes
    # nothing while the P pair blocks the other; with the P pair parked, it lets less by
    flipped = transmission(TREX, dict(flip_m2=True))
    assert np.isclose(flipped.transmitted, result.transmitted, rtol=1e-6)
    for mode in ('HF', 'HR'):
        free = transmission(TREX, dict(mode=mode), parked=('p1', 'p2'))
        turned = transmission(TREX, dict(mode=mode, flip_m2=True), parked=('p1', 'p2'))
        assert turned.transmitted < 0.8 * free.transmitted, mode
    # every allowed RRM turns every disc at a multiple of 14 Hz
    for rrm in RRM:
        assert all(abs(v) % FREQUENCY == 0 for v in speeds(rrm).values()), rrm
    try:
        TREX.choppers(rrm=6)
    except ValueError:
        pass
    else:
        raise AssertionError('RRM 6 accepted')
    print('ok')
