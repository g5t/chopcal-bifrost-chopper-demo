"""The ESS butterfly moderator as McStas's ``ESS_butterfly`` describes it, in absolute units.

A port of the brightness ``ESS_butterfly.comp`` (McStas 3) samples: Troels Schönfeldt's 2015
fits to the cold and thermal moderators' spectra, the time profile each wavelength has,
and the component's weighting by beamline -- emitting widths, intensity profiles across
the moderator's face, per-beamline performance scalars and the projection onto the beam.
Nothing here is sampled: for a beamline, a solid angle and bins in wavelength and emission
time, `intensity` gives the time-averaged rate into each bin, in neutrons per second, as
the total weight a McStas run of the component puts there.

A neutron of wavelength λ is emitted at time t after the proton pulse starts with density

    S(λ) T(t; τ(λ)) / D,   T = 1 - exp(-t/τ) while t < D,
                           T = (1 - exp(-D/τ)) exp(-(t - D)/τ) after,

with D the proton pulse's duration and τ a fitted decay time. T integrates to D exactly,
so S(λ) is the time-averaged brightness, n / (s cm² sr Å), at the reference power, and
holds per pulse whatever D is -- as McStas has it. The decay is an extrapolation past
the few pulse lengths the fits were made over.

Only the 3 cm high moderator, and not the Test Beamline's view (sector W, beamline 11).
Times in s, wavelengths in Å, lengths in m.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

#: The proton pulse's design duration, s, and repetition rate, Hz.
PULSE_DURATION = 2.857e-3
FREQUENCY = 14.0
#: The accelerator power the brightness is given at, MW.
REFERENCE_POWER = 5.0

#: Beamline axis angles, degrees, by sector; then the moderators' widths and the
#: performance scalars by beamline, from ESS_butterfly.comp.
_AXES = {'N': [30.0, 36.0, 42.0, 48.0, 54.0, 60.0, 66.0, 72.0, 78.0, 84.0, 90.0],
         'E': [-30.0, -36.0, -42.0, -48.0, -54.0, -60.0, -66.0, -72.0, -78.0, -84.0, -90.0],
         'W': [150.0, 144.7, 138.0, 132.7, 126.0, 120.7, 114.0, 108.7, 102.0, 96.7, 90.0],
         'S': [-150.0, -144.7, -138.0, -132.7, -126.0, -120.7, -114.0, -108.7, -102.0,
               -96.7, -90.0]}
_COLD_WIDTH = {'NE': [7e-2, 7.45e-2, 8.3e-2, 8.6e-2, 8.7e-2, 8.8e-2, 8.8e-2, 8.7e-2, 8.6e-2,
                      8.3e-2],
               'SW': [7e-2, 7.45e-2, 8.3e-2, 8.6e-2, 8.7e-2, 8.8e-2, 8.8e-2, 8.8e-2, 8.6e-2,
                      8.4e-2, 6.9e-2]}
_THERMAL_WIDTH = {'NE': [5.4e-2, 6.2e-2, 7.2e-2, 8.2e-2, 8.5e-2, 9.1e-2, 9.6e-2, 10e-2,
                         10.3e-2, 10.5e-2],
                  'SW': [5.4e-2, 6.2e-2, 7.2e-2, 8.2e-2, 8.5e-2, 9.1e-2, 9.6e-2, 9.95e-2,
                         10.25e-2, 10.45e-2, 10.5e-2]}
_COLD_SCALAR = {
    'N': [9.8788e-01, 1.0009e+00, 9.9335e-01, 9.5997e-01, 9.0717e-01, 9.1646e-01,
          9.1028e-01, 9.1773e-01, 9.2537e-01, 9.1727e-01],
    'E': [9.9032e-01, 1.0020e+00, 9.9647e-01, 9.6885e-01, 9.0713e-01, 9.1787e-01,
          9.1190e-01, 9.2113e-01, 9.2786e-01, 9.2146e-01],
    'W': [9.9017e-01, 1.0069e+00, 9.9366e-01, 9.7144e-01, 9.0624e-01, 8.9379e-01,
          9.1022e-01, 9.2847e-01, 9.2812e-01, 9.2703e-01, 8.3098e-01],
    'S': [8.6550e-01, 1.0071e+00, 9.9401e-01, 9.6243e-01, 9.0398e-01, 8.9299e-01,
          9.0830e-01, 9.2450e-01, 9.2270e-01, 9.2373e-01, 8.2508e-01]}
_THERMAL_SCALAR = {
    'N': [8.6782e-01, 7.8627e-01, 7.6528e-01, 7.9469e-01, 7.3645e-01, 7.3012e-01,
          7.2755e-01, 7.1750e-01, 7.1973e-01, 7.0459e-01],
    'E': [8.6838e-01, 7.8295e-01, 7.6719e-01, 7.9431e-01, 7.3989e-01, 7.3107e-01,
          7.2811e-01, 7.2201e-01, 7.2097e-01, 7.0307e-01],
    'W': [8.7232e-01, 8.0007e-01, 7.6853e-01, 8.0251e-01, 7.3728e-01, 7.3761e-01,
          7.2808e-01, 7.2151e-01, 7.1797e-01, 6.9857e-01, 6.9610e-01],
    'S': [8.6910e-01, 7.9964e-01, 7.6365e-01, 7.9922e-01, 7.3479e-01, 7.3836e-01,
          7.2773e-01, 7.2202e-01, 7.1667e-01, 7.0149e-01, 7.0084e-01]}
_DX_COLD = [-0.01, -0.01, -0.002, 0.004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
_DX_THERMAL = [0.002, 0.003, 0.002, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007]
#: The cold wing's face is turned this far from the thermal moderator's, degrees.
_WING = 24.24


# -- spectra and time profiles, from ESS_butterfly-lib.c ---------------------------------

def cold_spectrum(wavelength, angle):
    """The cold moderator's time-averaged brightness at 5 MW, n / (s cm² sr Å), seen at
    a beam port `angle` of 5, 15, ..., 55 degrees."""
    w = np.asarray(wavelength, dtype=float)
    par6 = {5: 5.73745e15, 15: 5.88284e15, 25: 6.09573e15, 35: 6.29116e15, 45: 6.03436e15,
            55: 6.02045e15}[angle] / 25
    i = (angle - 5) / 10
    par3, par5 = -13 - 0.5 * (angle - 5), -0.0478073 - 0.160 * np.exp(-0.45186 * i)
    par7, par8 = 0.788956 + 0.00854184 * i, 0.0461868 - 0.0016464 * i
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        para = ((1 + np.exp(par3 * (w - 2.53))) ** par5
                * par6 * (np.exp(-par7 * w) + par8 * np.exp(-0.325 * w)))
        ortho = 8.44e13 / 25 / ((1 + np.exp(2.5 * (w - 2.2))) * w)
        return np.where(w > 0, para + ortho, 0.0)


def thermal_spectrum(wavelength, angle):
    """The thermal moderator's time-averaged brightness at 5 MW, n / (s cm² sr Å)."""
    w = np.asarray(wavelength, dtype=float)
    i = (angle - 5) / 10
    par0 = 4.2906e13 - 9.2758e11 * i + 8.02603e11 * i * i - 1.29523e11 * i ** 3
    par2, par3 = 6.24806e12 - 8.84602e10 * i, -0.31107 + 0.0221138 * i
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        a = 949 / (325 * w * w)
        value = (par0 * 2 * a * a / w * w ** -par3 * np.exp(-a)
                 + par2 / ((1 + np.exp(2.5 * (w - 0.88))) * w))
        return np.where(w > 0, value, 0.0)


def cold_decay(wavelength, height=3.0):
    """The cold moderator's decay time, s; `height` in cm, which McStas fixes at 3."""
    w = np.asarray(wavelength, dtype=float)
    return (3.00094e-4 * (4.15681e-3 * w * w + 2.96212e-1 * np.exp(-1.78408e-1 * height)
                          + 7.77496e-1)
            * np.exp(-6.63537e1 * np.maximum(1e-13, w + 0.9) ** -8.64455))


def thermal_decay(wavelength, height=3.0):
    """The thermal moderator's decay time, s."""
    w = np.asarray(wavelength, dtype=float)
    return (3e-4 * (1.23048e-2 * w * w + 1.75628e-1 * np.exp(-1.82452e-1 * height)
                    + 9.27770e-1)
            * np.exp(-3.91090e1 * np.maximum(1e-13, w + 9.87990e-1) ** -7.65675))


def emitted_by(time, tau, duration=PULSE_DURATION):
    """The fraction of a pulse's neutrons emitted before `time`: T / D integrated from 0.

    Closed form, and safe for the vanishing decay times of short wavelengths."""
    t = np.maximum(np.asarray(time, dtype=float), 0.0)
    tau = np.maximum(np.asarray(tau, dtype=float), 1e-300)
    during = t - tau * -np.expm1(-np.minimum(t, duration) / tau)
    after = duration - tau * -np.expm1(-duration / tau) * np.exp(
        -np.maximum(t - duration, 0.0) / tau)
    return np.where(t < duration, during, after) / duration


# -- the profiles across the moderators' faces, 3 cm moderator --------------------------

def _cold_y(y):
    sinh_ish = 1 / ((1 + np.exp(30 * (y - 1.5))) * (1 + np.exp(-30 * (y + 1.5))))
    return 0.5 * (np.exp(-0.35 * y) + np.exp(0.35 * y)) * sinh_ish


def _thermal_y(y):
    edge = 1.5 - 0.105
    return 1.005 * np.where(np.abs(y) > edge, np.exp(-((np.abs(y) - edge) / 0.372) ** 2), 1.0)


def _cold_x(x, angle, width):
    i = (angle - 5) / 10
    par0 = 0.0146115 + 0.00797729 * i - 0.00279541 * i * i
    par1 = {1: 0.974217, 2: 0.981462, 3: 1.01466, 4: 1.11707, 5: 1.16057}.get(i, 0.980886)
    par2 = -20 if i == 0 else -4 - 0.75 * i
    par3 = (-14.9402 - 0.178369 * i + 0.0367007 * i * i) * (0.95 if i == 0 else 1)
    par4 = {3: -3.5, 5: -1.9}.get(i, -15)
    par5 = (-7.07979 + 0.0835695 * i - 0.0546662 * i * i) * (0.85 if i == 5 else 1)
    x = x * (par5 - par3) / width - 7.16
    cut = 1 / ((1 + np.exp(par2 * (x - par3))) * (1 + np.exp(-par4 * (x - par5))))
    return (par0 * (x + 12) + par1) * cut


def _thermal_x(x, angle, width):
    i = (angle - 5) / 10
    par0, par1 = -5.54775 + 0.492804 * i, -2.55 if angle == 55 else -0.265929 - 0.711477 * i
    par2, par3 = 0.821885 + 0.00914832 * i, 1.23 if angle == 55 else 1.31108 - 0.00698647 * i
    par4, par5 = -0.035, -0.0817358 + 0.00807125 * i
    par7 = {45: -8.2, 55: -7.7}.get(angle, -7.15)
    par9 = {45: 7.5, 55: 8.2}.get(angle, 7.15)
    x = x * (par9 - par7) / width - 7.16
    soften1, soften2 = 1 / (1 + np.exp(8 * (x - par0))), 1 / (1 + np.exp(8 * (x - par1)))
    cut = 1 / ((1 + np.exp(-8 * (x - par7))) * (1 + np.exp(8 * (x - par9))))
    line1 = par4 * (x - par0) + par2
    line2 = (par2 - par3) / (par0 - par1) * (x - par0) + par2
    line3 = par5 * (x - par1) + par3
    return cut * (line1 * soften1 + line2 * soften2 * (1 - soften1) + line3 * (1 - soften2))


# -- a beamline's view -------------------------------------------------------------------

@dataclass(frozen=True)
class Beamline:
    """What one beam port sees of the moderators: ``Beamline('W', 4)`` is BIFROST."""
    sector: str
    number: int
    height: float = 0.03   # m

    def __post_init__(self):
        if self.sector not in _AXES or not 1 <= self.number <= len(_AXES[self.sector]):
            raise ValueError(f'no beamline {self.sector}{self.number}')
        if self.height != 0.03:
            raise NotImplementedError('only the 3 cm high moderator')

    @property
    def _internal(self):
        return abs(90 - abs(_AXES[self.sector][self.number - 1]))

    @property
    def angle(self) -> int:
        """The beam port angle the fits are made at, nearest the beamline's own."""
        return min((5, 15, 25, 35, 45, 55), key=lambda a: abs(a - self._internal))

    @cached_property
    def areas(self) -> tuple[float, float]:
        """The cold and thermal moderators' effective emitting areas, cm²: what multiplies
        each one's brightness and the solid angle to give a rate.

        As ESS_butterfly weights a neutron: the nominal area (7.2 or 10.8 cm wide), the
        mean of the intensity profile over the face it samples, the beamline's scalar,
        and the cosine of the face's angle to the beam.
        """
        pair = 'NE' if self.sector in 'NE' else 'SW'
        n = self.number - 1
        cos_thermal = np.cos(np.radians(self._internal))
        cos_cold = np.cos(np.radians(self._internal - _WING))
        y = 100 * self.height * (np.arange(400) + 0.5) / 400 - 50 * self.height   # cm
        s = (np.arange(2000) + 0.5) / 2000   # along the sampled face, 0 to 1
        # cold: one face, running away from the beam axis in the moderator's own frame
        cold_length = _COLD_WIDTH[pair][n] / cos_cold
        x = 100 * (-cold_length * np.cos(np.radians(_WING)) * s - _DX_COLD[n])
        cold_profile = (_cold_x(x, self.angle, 100 * cold_length).mean()
                        * _cold_y(y).mean())
        # thermal: the broad face, and for beamlines 1-3 a little of the other side
        thermal_length = 1.1 * _THERMAL_WIDTH[pair][n] / cos_thermal
        broad = 1.0 if self.number >= 4 else 1 - 0.072
        width = (100 * _THERMAL_WIDTH[pair][n] + 0.7) / cos_thermal
        profile = 0.0
        for share, sign in ((broad, 1), (1 - broad, -1)):
            if share:
                x = 100 * (sign * share * thermal_length * s + _DX_THERMAL[n])
                profile += share * _thermal_x(x, self.angle, width).mean()
        thermal_profile = profile * _thermal_y(y).mean()
        cold = (720 * self.height * cold_profile * _COLD_SCALAR[self.sector][n]
                * cos_cold)
        thermal = (1080 * self.height * thermal_profile * _THERMAL_SCALAR[self.sector][n]
                   * cos_thermal)
        return float(cold), float(thermal)

    def intensity(self, wavelength_edges, time_edges, solid_angle, power=REFERENCE_POWER,
                  duration=PULSE_DURATION, decay_scale=1.0, cold=1.0, thermal=1.0,
                  points=4):
        """The time-averaged rate into each (emission time, wavelength) bin, n/s, through
        `solid_angle` (sr) about the beamline's axis; shaped (time, wavelength).

        `power` (MW) scales everything; `duration` is the proton pulse's (s);
        `decay_scale` multiplies both decay times; `cold` and `thermal` are
        ESS_butterfly's performance multipliers. Each wavelength bin is integrated with
        `points` Gauss-Legendre points, each time bin exactly. A rate per pulse is this
        over `FREQUENCY`.
        """
        if self.sector == 'W' and self.number == 11:
            raise NotImplementedError("the Test Beamline's view of both cold wings")
        w = np.asarray(wavelength_edges, dtype=float)
        t = np.asarray(time_edges, dtype=float)
        nodes, weights = np.polynomial.legendre.leggauss(points)
        half = 0.5 * np.diff(w)
        at = (0.5 * (w[1:] + w[:-1]))[:, None] + half[:, None] * nodes   # (bins, points)
        cold_area, thermal_area = self.areas
        total = np.zeros((len(t) - 1, len(w) - 1))
        for area, factor, spectrum, decay in (
                (cold_area, cold, cold_spectrum, cold_decay),
                (thermal_area, thermal, thermal_spectrum, thermal_decay)):
            tau = decay_scale * decay(at)
            fraction = np.diff(emitted_by(t[:, None, None], tau, duration), axis=0)
            total += area * factor * np.einsum(
                'tbp,bp,p,b->tb', fraction, spectrum(at, self.angle), weights, half)
        return total * solid_angle * power / REFERENCE_POWER


if __name__ == '__main__':
    bifrost = Beamline('W', 4)
    assert bifrost.angle == 45
    # every pulse's neutrons are emitted, whatever its length or decay
    for duration in (PULSE_DURATION, 1e-3):
        for tau in (1e-30, 1e-4, 4e-4):
            assert np.isclose(emitted_by(1.0, tau, duration), 1.0)
            assert np.isclose(emitted_by(duration, tau, duration),
                              1 - tau / duration * -np.expm1(-duration / tau))
    # bins add up: finer time bins sum to the coarse ones
    w = np.array([2.0, 4.0, 6.0])
    coarse = bifrost.intensity(w, [0.0, 3e-3, 9e-3], 1e-3)
    fine = bifrost.intensity(w, np.linspace(0.0, 9e-3, 31), 1e-3)
    assert np.allclose(coarse.sum(axis=0), fine.sum(axis=0))
    # McStas ESS_butterfly (sector W, beamline 4, 5 MW, emission to 3 pulse lengths)
    # into BIFROST's guide entrance, 68.70 x 34.45 mm at 1.9034 m, in 0.01 Å bands:
    # guide_entrance_I from bifrost-guide-transmission's scan, 1e6 rays a point, ~0.3 %
    solid_angle = 0.068701 * 0.034454 / 1.90343 ** 2
    for wavelength, mcstas in ((1.0, 5.49234e9), (2.5, 5.39517e9), (5.0, 8.86362e8),
                               (10.0, 6.49468e7)):
        edges = [wavelength - 0.005, wavelength + 0.005]
        ours = bifrost.intensity(edges, [0.0, 3 * PULSE_DURATION], solid_angle).sum()
        assert abs(ours / mcstas - 1) < 0.01, (wavelength, ours / mcstas)
    print('ok')
