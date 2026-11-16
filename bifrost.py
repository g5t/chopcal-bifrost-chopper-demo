"""BIFROST, for `phasespace`: beamline W4, six discs set by `chopcal.bifrost`.

    import bifrost, phasespace
    phasespace.figure(bifrost.BIFROST, energy_min=4.5, shaping_time=0.2)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import ess_butterfly
from phasespace import ESS, Instrument

#: The fastest the designed pulse-shaping disks may turn, Hz: the 15th harmonic of 14 Hz.
#: chopcal's own default is 196 Hz. The 126 Hz substitutes installed now are the 9th.
PSC_SPEED_MAX = 210.0


class Bifrost(Instrument):
    name = 'BIFROST'
    source = ESS(ess_butterfly.Beamline('W', 4))
    #: the guide's entrance, 68.70 x 34.45 mm at 1.9034 m from the moderator (niess)
    solid_angle = 0.068701 * 0.034454 / 1.90343 ** 2
    sample = 162.0
    names = ('ps1', 'ps2', 'fo1', 'fo2', 'bw1', 'bw2')
    groups = (0, 0, 1, 2, 3, 3)   # the pulse-shaping and the bandwidth pairs share housings
    #: what chopcal.bifrost gives over the controls' ranges, and some
    delay_ranges_ms = {'ps1': (-20.0, 20.0), 'ps2': (0.0, 40.0), 'fo1': (0.0, 25.0),
                       'fo2': (0.0, 40.0), 'bw1': (0.0, 180.0), 'bw2': (0.0, 180.0)}
    #: what reaches a 10 x 30 mm sample window over what enters the guide, choppers
    #: removed, from McStas: bifrost-guide-transmission (niess's BIFROST, with restage)
    guide_table = Path(__file__).with_name('bifrost_guide_transmission.csv')
    wavelength_range = (1.0, 6.0)
    latest_emission = 6e-3

    def choppers(self, energy_min=4.5, shaping_time=2e-4,
                 pulse_shaping_speed_max=PSC_SPEED_MAX):
        """chopcal's settings for the lowest energy (meV) and pulse-shaping time (s)."""
        import chopcal
        from chopcal import constants as c
        from chopcal.lib import beam_aperture
        settings = chopcal.bifrost(energy_min=energy_min, shaping_time=shaping_time,
                                   pulse_shaping_speed_max=pulse_shaping_speed_max)
        narrow = beam_aperture(c.DISK_RADIUS, c.NARROW_SLIT_HEIGHT, c.NARROW_WINDOW_WIDTH,
                               c.NARROW_WINDOW_HEIGHT, at='centre')
        wide = beam_aperture(c.DISK_RADIUS, c.BANDWIDTH_SLIT_HEIGHT,
                             c.BANDWIDTH_WINDOW_WIDTH, c.BANDWIDTH_WINDOW_HEIGHT, at='centre')
        return [(settings[n], wide if n.startswith('bw') else narrow) for n in self.names]

    def controls(self, energy_min=4.5, shaping_time=0.2, psc_speed_max=PSC_SPEED_MAX):
        from ipywidgets import FloatSlider, SelectionSlider
        return {
            'energy_min': FloatSlider(value=energy_min, min=1.0, max=20.0, step=0.1,
                                      description='lowest energy [meV]'),
            'shaping_time': FloatSlider(value=shaping_time, min=0.05, max=4.0, step=0.05,
                                        description='shaping time [ms]'),
            # the disks turn at a harmonic of 14 Hz, so their limit is one too
            'psc_speed_max': SelectionSlider(options=[(f'{14 * n} Hz', 14.0 * n)
                                                      for n in range(1, 16)],
                                             value=psc_speed_max, description='PSC limit'),
        }

    def settings(self, values):
        return dict(energy_min=values['energy_min'],
                    shaping_time=values['shaping_time'] * 1e-3,
                    pulse_shaping_speed_max=values['psc_speed_max'])

    def note(self, result, values):
        """The speed chopcal picked for the pulse-shaping pair: the fastest harmonic the
        burst allows."""
        speed = abs(result.speeds[0] or result.speeds[1])
        if not speed:
            return 'PSC parked'
        slower = (' -- slower than the limit, for a burst this long'
                  if speed < values['psc_speed_max'] else '')
        return f'PSC turns at {speed:g} Hz{slower}'


BIFROST = Bifrost()


if __name__ == '__main__':
    import chopcal
    from chopcal.lib import wavelength_to_inverse_velocity
    from phasespace import _figure, transmission

    def run(**kwargs):
        settings = {k: kwargs.pop(k) for k in ('energy_min', 'shaping_time') if k in kwargs}
        return transmission(BIFROST, settings, **{'latest_emission': 3e-3, **kwargs})

    # the stages end with exactly what the train transmits, the outlined region
    result = run()
    assert len(result.stages[-1]) == len(result.outline)
    last = np.array(result.stages[-1][0])
    assert np.isclose(last[:, 0].min(), min(t for t, _ in result.outline[0]))
    # what the train transmits crosses every disc inside one of its openings
    per_angstrom = wavelength_to_inverse_velocity(1.0)
    for (_, at), windows in zip(result.positions, result.openings):
        for polygon in result.outline:
            times = [t + at * w * per_angstrom for t, w in polygon]
            assert any(a - 1e-9 <= min(times) and max(times) <= b + 1e-9
                       for a, b in windows), at
    # the guide only removes neutrons, least at long wavelengths
    guide = BIFROST.guide_transmission
    assert np.all(guide([0.5, 2.0, 12.0, 30.0]) <= 1.0)
    assert guide(1.0) < 0.005 < 0.05 < guide(10.0)
    assert result.transmitted_guide < 0.1 * result.transmitted
    # by wavelength, the guide multiplies each bin: the profile sums to the guided total
    assert np.isclose((result.by_wavelength * result.guide).sum(), result.transmitted_guide)
    # replacing every delay by chopcal's plus x moves the train as a pulse delay x does
    settings = chopcal.bifrost(energy_min=4.5, shaping_time=2e-4,
                               pulse_shaping_speed_max=PSC_SPEED_MAX)
    moved = run(delays={n: settings[n].delay + 1e-4 for n in BIFROST.names})
    assert np.allclose(moved.outline, run(pulse_delay=1e-4).outline)
    # a long emission window opens a second, spurious region, far weaker than the
    # intended one, which is emitted during the pulse; its parts are within the totals
    late = run(latest_emission=15e-3, wavelength_range=(1.0, 8.0))
    assert late.main.count(False) >= 1 and result.main == [True] * len(result.outline)
    assert 0 < late.transmitted_spurious < 1e-6 * late.transmitted
    assert np.isclose(late.at_sample_spurious.sum(), late.transmitted_spurious)
    assert np.isclose(late.transmitted - late.transmitted_spurious, result.transmitted,
                      rtol=0.02)
    assert all(min(t for t, _ in late.outline[i]) > 5e-3
               for i, main in enumerate(late.main) if not main)
    # the source knobs: power scales everything; at the same current, half the pulse is
    # half the neutrons, once the window holds all of both pulses
    assert np.isclose(run(power=2.5).transmitted, 0.5 * result.transmitted)
    whole = run(latest_emission=20e-3).emitted
    half = run(latest_emission=20e-3, pulse_duration=ess_butterfly.PULSE_DURATION / 2).emitted
    assert np.isclose(half, 0.5 * whole, rtol=1e-3), half / whole
    # a parked disc passes everything: parking every disc passes the whole rectangle,
    # and parking one never passes less
    everything = run(parked=BIFROST.names)
    assert np.isclose(everything.transmitted, everything.emitted)
    assert everything.speeds == [0.0] * 6 and len(everything.openings[0]) == 1
    for name in BIFROST.names:
        assert run(parked=(name,)).transmitted >= result.transmitted * (1 - 1e-9)
    assert run(parked=('ps1', 'ps2')).transmitted > 2 * result.transmitted

    # the delay sliders follow chopcal, and keep a hand-set value until one of chopcal's
    # inputs changes; parking a disc greys out its delay
    import matplotlib
    matplotlib.use('module://ipympl.backend_nbagg')
    panel = _figure(BIFROST).children[1].children
    control = {w.description: w for w in panel if hasattr(w, 'description')}
    rows = {r.children[0].description.split()[0]: r.children for r in panel
            if type(r).__name__ == 'HBox'}

    def chopcal_ms(name):
        return chopcal.bifrost(energy_min=control['lowest energy [meV]'].value,
                               shaping_time=control['shaping time [ms]'].value * 1e-3,
                               pulse_shaping_speed_max=control['PSC limit'].value
                               )[name].delay * 1e3

    assert all(np.isclose(rows[n][0].value, chopcal_ms(n)) for n in BIFROST.names)
    rows['ps1'][0].value = 1.234                   # by hand
    control['last emission [ms]'].value = 4.0      # not one of chopcal's inputs
    assert rows['ps1'][0].value == 1.234
    control['lowest energy [meV]'].value = 7.0     # one of chopcal's inputs
    assert all(np.isclose(rows[n][0].value, chopcal_ms(n)) for n in BIFROST.names)
    rows['fo1'][1].value = True                    # parked
    assert rows['fo1'][0].disabled and not rows['fo2'][0].disabled
    print('ok')
