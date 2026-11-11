"""BIFROST's source intensity through its chopper train, from chopcal's exact regions.

The ESS source BIFROST sees is tabulated by ``tof`` on a fine grid of (wavelength,
emission time). chopcal builds the region of that plane the six choppers transmit --
exactly, as convex polygons -- and weights it by how likely each neutron is to get
through a disc whose edge is still sweeping the beam. Integrating the gridded source over
that weighted region gives what reaches the sample, by arrival time and by wavelength.

Three layers, so the same numbers serve a notebook and a static picture:

- `transmission` computes everything for one setting, with no plotting;
- `draw` puts a result on three matplotlib axes;
- `bifrost_transmitted_phasespace_figure` wires both to ipywidgets controls.

Needs chopcal >= 0.7 (`Region.transmit_weighted`, `beam_aperture(..., at='centre')`) and
``tof`` for the source distribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

#: The ESS source distributions in ``tof`` are sampled this far from the moderator, and
#: their emission times are counted there; chopcal measures its paths from the moderator.
SOURCE_PLANE = 0.05  # m
#: The pulse-shaping pair and the bandwidth pair each share a housing.
GROUPS = [0, 0, 1, 2, 3, 3]
NAMES = ('ps1', 'ps2', 'fo1', 'fo2', 'bw1', 'bw2')


@lru_cache(maxsize=4)
def source_distribution(name: str = 'ess-bifrost'):
    """The tabulated source, (wavelength, birth_time), as amounts on a fine uniform grid.

    Read with bin-edge coordinates so that it can be rebinned: an amount per cell, spread
    evenly over it, which is how chopcal reads a gridded source too.
    """
    import scipp as sc
    from tof.facilities import get_source_path
    data = sc.io.load_hdf5(get_source_path(name))
    for dim in ('wavelength', 'birth_time'):
        centres = data.coords[dim]
        step = centres[1] - centres[0]
        data.coords[dim] = sc.concat([centres - 0.5 * step, centres[-1:] + 0.5 * step], dim)
    data.coords['birth_time'] = data.coords['birth_time'].to(unit='s')
    return data


def rebinned(wavelength_edges, time_edges, name: str = 'ess-bifrost') -> np.ndarray:
    """The source summed into these bins, shaped (time, wavelength) as chopcal wants it."""
    import scipp as sc
    data = source_distribution(name)
    out = data.rebin(wavelength=sc.array(dims=['wavelength'], values=wavelength_edges,
                                         unit='angstrom'),
                     birth_time=sc.array(dims=['birth_time'], values=time_edges, unit='s'))
    return np.ascontiguousarray(out.transpose(['birth_time', 'wavelength']).values)


def trains(energy_min: float, shaping_time: float, pulse_shaping_speed_max: float = 196.0,
           pulse_delay: float = 0.0):
    """BIFROST's discs twice over, with paths from the source plane rather than the
    moderator: once with the beam's inner-corner aperture, which bounds where anything
    gets through, and once with its centre aperture, which sets the ramps a weighted
    transmission models.

    `pulse_delay` (s) is added to every disc's delay, as ESS's ``BeamPosDly-S`` is when
    the experiment delays NICOS sets stay as they are: the whole train moves together.
    chopcal's settings are for none.
    """
    import chopcal
    from chopcal import constants as c
    from chopcal.lib import Chopper, beam_aperture
    settings = chopcal.bifrost(energy_min=energy_min, shaping_time=shaping_time,
                               pulse_shaping_speed_max=pulse_shaping_speed_max)
    centre = {
        'narrow': beam_aperture(c.DISK_RADIUS, c.NARROW_SLIT_HEIGHT, c.NARROW_WINDOW_WIDTH,
                                c.NARROW_WINDOW_HEIGHT, at='centre'),
        'wide': beam_aperture(c.DISK_RADIUS, c.BANDWIDTH_SLIT_HEIGHT,
                              c.BANDWIDTH_WINDOW_WIDTH, c.BANDWIDTH_WINDOW_HEIGHT,
                              at='centre'),
    }

    def moved(disc, aperture):
        return Chopper(disc.speed, disc.delay + pulse_delay, disc.beam, list(disc.edges),
                       disc.path - SOURCE_PLANE, aperture)

    support = [moved(settings[n], settings[n].aperture) for n in NAMES]
    ramped = [moved(settings[n], centre['wide' if n.startswith('bw') else 'narrow'])
              for n in NAMES]
    return support, ramped


@dataclass
class Transmission:
    """Everything one setting produces, in plain arrays."""
    wavelength_edges: np.ndarray        # Å
    time_edges: np.ndarray              # s, at the source plane
    source: np.ndarray                  # (time, wavelength) amount per bin
    outline: list                       # transmitted polygons as [(t, λ), ...]
    arrival_edges: np.ndarray           # s, at the sample
    at_sample: np.ndarray               # weighted amount per arrival bin
    at_sample_support: np.ndarray       # the same counting the whole region as open
    by_wavelength: np.ndarray           # weighted amount per wavelength bin
    by_wavelength_support: np.ndarray
    transmitted: float                  # weighted total
    emitted: float                      # total in the source rectangle
    psc_speed: float                    # Hz, what the pulse-shaping pair turns at


def transmission(energy_min=4.5, shaping_time=2e-4, wavelength_range=(1.0, 6.0),
                 latest_emission=3e-3, wavelength_bin=0.02, time_bin=20e-6,
                 arrival_bins=400, pulse_shaping_speed_max=196.0, pulse_delay=0.0,
                 source='ess-bifrost') -> Transmission:
    """The source in a (wavelength, emission time) rectangle, through BIFROST's train.

    The rectangle is the source polygon chopcal transmits and the extent of the gridded
    source; the bin sizes set the grid. `pulse_delay` moves every disc together; see
    `trains`. Times in s, wavelengths in Å.
    """
    from chopcal import constants as c
    from chopcal.lib import Region, wavelength_to_inverse_velocity

    low, high = sorted(wavelength_range)
    wavelength_edges = np.arange(low, high + 0.5 * wavelength_bin, wavelength_bin)
    time_edges = np.arange(0.0, latest_emission + 0.5 * time_bin, time_bin)
    amounts = rebinned(wavelength_edges, time_edges, source)

    per_angstrom = wavelength_to_inverse_velocity(1.0)   # chopper-lib's own conversion
    inverse_velocities = wavelength_edges * per_angstrom
    rectangle = Region.rectangle(inverse_velocities[0],
                                 inverse_velocities[-1] - inverse_velocities[0],
                                 time_edges[0], time_edges[-1] - time_edges[0])
    support_train, ramped_train = trains(energy_min, shaping_time, pulse_shaping_speed_max,
                                         pulse_delay)
    support = rectangle.transmit(support_train)
    weighted = rectangle.transmit_weighted(ramped_train, GROUPS)

    distance = c.INSTRUMENT_LENGTH - SOURCE_PLANE
    if support:
        reached = [t + distance * u for p in support.polygons for u, t in p.vertices]
        first, last = min(reached), max(reached)
        margin = 0.02 * (last - first)
        arrival_edges = np.linspace(first - margin, last + margin, arrival_bins + 1)
    else:
        arrival_edges = np.linspace(0.0, 1 / 14, arrival_bins + 1)

    def arrivals(region):
        return region.arrivals(amounts, inverse_velocities, time_edges, distance,
                               arrival_edges).sum(axis=1)

    def by_wavelength(region):
        return (amounts * region.coverage(inverse_velocities, time_edges)).sum(axis=0)

    outline = [[(t, u / per_angstrom) for u, t in p.vertices] for p in support.polygons]
    return Transmission(
        wavelength_edges=wavelength_edges, time_edges=time_edges, source=amounts,
        outline=outline, arrival_edges=arrival_edges,
        at_sample=arrivals(weighted), at_sample_support=arrivals(support),
        by_wavelength=by_wavelength(weighted), by_wavelength_support=by_wavelength(support),
        transmitted=weighted.integrate(amounts, inverse_velocities, time_edges),
        emitted=float(amounts.sum()), psc_speed=abs(support_train[0].speed))


# -- wavelength and energy --------------------------------------------------------------

def _energy(wavelength):
    """meV from Å, through chopcal's h^2/2m, safe at the zero matplotlib probes with."""
    from chopcal import constants as c
    w = np.asarray(wavelength, dtype=float)
    with np.errstate(divide='ignore'):
        return np.where(w > 0, c.H2_OVER_2M / np.where(w > 0, w, 1.0) ** 2, np.inf)


def _wavelength(energy):
    from chopcal import constants as c
    e = np.asarray(energy, dtype=float)
    with np.errstate(divide='ignore'):
        return np.where(e > 0, np.sqrt(c.H2_OVER_2M / np.where(e > 0, e, 1.0)), np.inf)


#: Energies worth a tick, meV: round numbers, sparse where the scale compresses them.
ENERGY_TICKS = (1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 50, 80, 120, 200)


def _energy_axis(secondary, wavelengths):
    """Label an energy axis twinned to a wavelength one, at round energies only."""
    low, high = sorted(_energy(np.asarray(wavelengths, dtype=float)))
    ticks = [e for e in ENERGY_TICKS if low <= e <= high]
    secondary.set_ticks(ticks, labels=[f'{e:g}' for e in ticks])
    return secondary


# -- drawing ----------------------------------------------------------------------------

def make_axes(figsize=(12, 6)):
    """A figure with the source plane on the left and two profiles stacked on the right."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=figsize, layout='constrained')
    grid = fig.add_gridspec(2, 2, width_ratios=(1.15, 1))
    left = fig.add_subplot(grid[:, 0])
    time = fig.add_subplot(grid[0, 1])
    wavelength = fig.add_subplot(grid[1, 1])
    return fig, (left, time, wavelength)


def draw(result: Transmission, axes, title: str | None = None, colorbar=None,
         log: bool = False):
    """Draw one result on `make_axes`'s three axes, replacing what was there.

    The source is on a linear colour scale, which shows the pulse's shape; `log` shows its
    reach instead. Returns the colorbar, so that a caller redrawing the same figure can
    hand it back instead of stacking a new one each time.
    """
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.patches import Polygon
    left, time, wavelength = axes
    for ax in axes:
        ax.clear()
        for secondary in list(getattr(ax, 'child_axes', [])):
            secondary.remove()

    w, t = result.wavelength_edges, result.time_edges
    density = result.source / np.outer(np.diff(t * 1e3), np.diff(w))   # per ms per Å
    positive = density[density > 0]
    if not positive.size:
        norm = None
    elif log:
        norm = LogNorm(vmin=positive.max() * 1e-4, vmax=positive.max())
    else:
        norm = Normalize(vmin=0.0, vmax=positive.max())
    mesh = left.pcolormesh(t * 1e3, w, density.T, norm=norm, cmap='viridis',
                           shading='flat', rasterized=True)
    for polygon in result.outline:
        left.add_patch(Polygon([(a * 1e3, b) for a, b in polygon], closed=True,
                               fill=False, edgecolor='white', linewidth=1.6))
        left.add_patch(Polygon([(a * 1e3, b) for a, b in polygon], closed=True,
                               fill=False, edgecolor='crimson', linewidth=0.9))
    left.set_xlim(t[0] * 1e3, t[-1] * 1e3)
    left.set_ylim(w[0], w[-1])
    left.set_xlabel('emission time at the source [ms]')
    left.set_ylabel('wavelength [Å]')
    energy = left.secondary_yaxis('right', functions=(_energy, _wavelength))
    _energy_axis(energy.yaxis, (w[0], w[-1]))
    energy.set_ylabel('energy [meV]')
    left.set_title('source, and the region the train transmits')
    if colorbar is None:
        colorbar = left.figure.colorbar(mesh, ax=left, location='bottom', shrink=0.8,
                                        label='source intensity [per ms per Å, arb.]')
    else:
        colorbar.update_normal(mesh)

    a = result.arrival_edges * 1e3
    step = np.diff(a)
    time.stairs(result.at_sample_support / step, a, color='0.6', linestyle='--',
                label='whole region open')
    time.stairs(result.at_sample / step, a, color='C0', linewidth=1.5,
                label='weighted by transmission')
    time.set_xlabel('arrival time at the sample [ms]')
    time.set_ylabel('intensity [per ms, arb.]')
    time.set_xlim(a[0], a[-1])
    time.legend(loc='upper right', fontsize='small')

    wstep = np.diff(w)
    wavelength.stairs(result.by_wavelength_support / wstep, w, color='0.6', linestyle='--',
                      label='whole region open')
    wavelength.stairs(result.by_wavelength / wstep, w, color='C1', linewidth=1.5,
                      label='weighted by transmission')
    reached = np.nonzero(result.by_wavelength_support)[0]
    if reached.size:
        lo, hi = w[max(reached[0] - 2, 0)], w[min(reached[-1] + 3, len(w) - 1)]
        wavelength.set_xlim(lo, hi)
    wavelength.set_xlabel('wavelength [Å]')
    wavelength.set_ylabel('intensity [per Å, arb.]')
    top = wavelength.secondary_xaxis('top', functions=(_energy, _wavelength))
    _energy_axis(top.xaxis, wavelength.get_xlim())
    top.set_xlabel('energy [meV]')
    wavelength.legend(loc='upper right', fontsize='small')

    fraction = result.transmitted / result.emitted if result.emitted else 0.0
    text = title + '\n' if title else ''
    time.set_title(f'{text}{fraction:.3%} of the source rectangle reaches the sample',
                   fontsize='medium')
    return colorbar


# -- interactive ------------------------------------------------------------------------

_IPYMPL = {'widget', 'ipympl', 'module://ipympl.backend_nbagg'}


def ipympl_active() -> bool:
    """True if ipympl is the current backend. No side effects, no resolution."""
    import matplotlib
    try:
        name = matplotlib.get_backend(auto_select=False)   # mpl >= 3.9
    except TypeError:                                       # older matplotlib
        from matplotlib import rcsetup
        name = dict.__getitem__(matplotlib.rcParams, 'backend')
        if name is rcsetup._auto_backend_sentinel:
            name = None
    return name is not None and name.lower() in _IPYMPL


def ensure_widget_backend(quiet: bool = False) -> bool:
    """Enable %matplotlib widget if we're in IPython and it isn't already on."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    ip = get_ipython()
    if ip is None:
        return False
    if ipympl_active():
        return True
    ip.run_line_magic('matplotlib', 'widget')
    if not quiet:
        print('[polygonfigure] enabled %matplotlib widget')
    return True


def _bifrost_transmitted_phasespace_figure(ei_min=4.5, t_psc=0.2):
    from ipywidgets import (Checkbox, FloatRangeSlider, FloatSlider, HBox, Label, Layout,
                            ToggleButtons, VBox)
    import matplotlib.pyplot as plt

    style = {'description_width': '9em'}
    wide = Layout(width='26em')
    controls = {
        'energy_min': FloatSlider(value=ei_min, min=1.0, max=20.0, step=0.1,
                                  description='lowest energy [meV]'),
        'shaping_time': FloatSlider(value=t_psc, min=0.05, max=2.0, step=0.05,
                                    description='shaping time [ms]'),
        'latest_emission': FloatSlider(value=3.0, min=0.5, max=6.0, step=0.1,
                                       description='last emission [ms]'),
        'wavelength_range': FloatRangeSlider(value=(1.0, 6.0), min=0.2, max=19.9, step=0.1,
                                             description='wavelengths [Å]'),
        'time_bin': FloatSlider(value=20.0, min=6.0, max=200.0, step=2.0,
                                description='time bin [µs]'),
        'wavelength_bin': FloatSlider(value=0.02, min=0.02, max=0.5, step=0.01,
                                      description='wavelength bin [Å]'),
        'pulse_delay': FloatSlider(value=0.0, min=-1000.0, max=1000.0, step=10.0,
                                   description='pulse delay [µs]'),
        'psc_speed_max': ToggleButtons(options=[('designed, 196 Hz', 196.0),
                                                ('substitute, 126 Hz', 126.0)],
                                       value=196.0, description='PSC disks'),
        'log': Checkbox(value=False, description='log colour scale'),
    }
    for widget in controls.values():
        widget.style, widget.layout, widget.continuous_update = style, wide, False
    controls['psc_speed_max'].style.button_width = '10em'

    with plt.ioff():
        fig, axes = make_axes()
    status = Label()
    state = {'colorbar': None}

    def update(*_):
        v = {k: w.value for k, w in controls.items()}
        result = transmission(energy_min=v['energy_min'], shaping_time=v['shaping_time'] * 1e-3,
                              wavelength_range=v['wavelength_range'],
                              latest_emission=v['latest_emission'] * 1e-3,
                              wavelength_bin=v['wavelength_bin'],
                              time_bin=v['time_bin'] * 1e-6,
                              pulse_delay=v['pulse_delay'] * 1e-6,
                              pulse_shaping_speed_max=v['psc_speed_max'])
        state['colorbar'] = draw(result, axes, colorbar=state['colorbar'], log=v['log'])
        fig.canvas.draw_idle()
        status.value = (f'{result.source.shape[1]} x {result.source.shape[0]} bins; '
                        f'{len(result.outline)} transmitted polygon(s); PSC at '
                        f'{result.psc_speed:g} Hz')

    for widget in controls.values():
        widget.observe(update, names='value')
    update()
    return HBox([fig.canvas, VBox([*controls.values(), status])])


def bifrost_transmitted_phasespace_figure(ei_min=4.5, t_psc=0.2):
    """The interactive figure: the source and the transmitted region on the left, what
    reaches the sample by arrival time and by wavelength on the right.

    `ei_min` is the lowest incident energy chopcal places the band at, in meV, and `t_psc`
    the pulse-shaping burst, in ms.
    """
    if ensure_widget_backend():
        return _bifrost_transmitted_phasespace_figure(ei_min=ei_min, t_psc=t_psc)
    raise RuntimeError('Interactive mode needed -- please install IPython and ipympl')
