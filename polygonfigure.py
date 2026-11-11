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
from itertools import combinations
from math import ceil, floor

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
    #: (name, distance from the moderator in m) for each chopper, then the sample
    positions: list
    #: for each entry of `positions`, the (time, distance) outlines of the neutrons that
    #: got through every chopper before it, swept from the source plane up to it
    stages: list
    #: for each chopper, the (start, end) times in s that any of the beam can pass it,
    #: from 0 to past the latest time in `stages`
    openings: list


def open_windows(chopper, until: float):
    """When a disc passes any of the beam, as sorted (start, end) times from 0 to `until`.

    An edge at angle a is on the beam at delay + (beam - a) / (360 speed), the openings
    recur every 1/|speed|, and the aperture widens each by half of itself at both ends.
    """
    turn = 360 * abs(chopper.speed)
    half = chopper.aperture / 2 / turn
    windows = []
    for a, b in chopper.openings:
        start, end = sorted(chopper.delay + (chopper.beam - x) / (360 * chopper.speed)
                            for x in (a, b))
        start, end = start - half, end + half
        for n in range(ceil(-end * turn / 360), floor((until - start) * turn / 360) + 1):
            windows.append((start + n * 360 / turn, end + n * 360 / turn))
    return sorted(windows)


def swept(vertices, l0: float, l1: float):
    """The (t, L) outline a polygon of (u, t) sweeps between distances l0 and l1.

    At distance L its neutrons span the times t + L u over its vertices. The latest and
    earliest of those are piecewise linear in L, bending only where two vertices swap, so
    evaluating at those crossings gives the outline exactly.
    """
    ls = {l0, l1}
    for (ua, ta), (ub, tb) in combinations(vertices, 2):
        if ua != ub and l0 < (tb - ta) / (ua - ub) < l1:
            ls.add((tb - ta) / (ua - ub))
    ls = sorted(ls)
    times = np.array([[t + l * u for u, t in vertices] for l in ls])
    return [*zip(times.max(axis=1), ls), *zip(times.min(axis=1)[::-1], ls[::-1])]


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

    # what gets through the first k choppers, swept from the source plane to the next one
    stops = [(n, d.path) for n, d in zip(NAMES, support_train)] + [('sample', distance)]
    stages, through = [], rectangle
    for k, (_, stop) in enumerate(stops):
        if k:
            through = through.transmit([support_train[k - 1]])
        stages.append([[(t, l + SOURCE_PLANE) for t, l in swept(p.vertices, 0.0, stop)]
                       for p in through.polygons])
    positions = [(n, stop + SOURCE_PLANE) for n, stop in stops]
    until = 1.1 * max((t for polygons in stages for q in polygons for t, _ in q),
                      default=1 / 14)
    openings = [open_windows(chopper, until) for chopper in support_train]
    return Transmission(
        wavelength_edges=wavelength_edges, time_edges=time_edges, source=amounts,
        outline=outline, arrival_edges=arrival_edges,
        at_sample=arrivals(weighted), at_sample_support=arrivals(support),
        by_wavelength=by_wavelength(weighted), by_wavelength_support=by_wavelength(support),
        transmitted=weighted.integrate(amounts, inverse_velocities, time_edges),
        emitted=float(amounts.sum()), psc_speed=abs(support_train[0].speed),
        positions=positions, stages=stages, openings=openings)


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

def make_axes(figsize=(12, 10)):
    """The source plane on the left and two profiles stacked on the right, above a
    time-distance diagram of the whole instrument and one of its first choppers."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=figsize, layout='constrained')
    grid = fig.add_gridspec(3, 2, width_ratios=(1.15, 1), height_ratios=(1, 1, 1.3))
    left = fig.add_subplot(grid[:2, 0])
    time = fig.add_subplot(grid[0, 1])
    wavelength = fig.add_subplot(grid[1, 1])
    distance = fig.add_subplot(grid[2, 0])
    near = fig.add_subplot(grid[2, 1])
    return fig, (left, time, wavelength, distance, near)


def draw(result: Transmission, axes, title: str | None = None, colorbar=None,
         log: bool = False):
    """Draw one result on `make_axes`'s axes, replacing what was there.

    The source and the profiles are on linear scales, which show the pulse's shape; `log`
    puts the colour scale and both profiles' intensities on log scales, which show its
    reach instead, down to 1e-4 of each one's peak. Returns the colorbar, so that a caller redrawing the same figure can
    hand it back instead of stacking a new one each time.
    """
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.patches import Polygon
    left, time, wavelength, *rest = axes
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
    if log:
        _log_intensity(time, result.at_sample_support / step)

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
    if log:
        _log_intensity(wavelength, result.by_wavelength_support / wstep)

    fraction = result.transmitted / result.emitted if result.emitted else 0.0
    text = title + '\n' if title else ''
    time.set_title(f'{text}{fraction:.3%} of the source rectangle reaches the sample',
                   fontsize='medium')
    # the whole instrument, then the choppers in its first quarter, where most is stopped
    near = [at for _, at in result.positions if at < 0.25 * result.positions[-1][1]]
    for ax, top in zip(rest, (None, 1.15 * max(near))):
        draw_time_distance(result, ax, top)
    return colorbar


def _latest_below(polygon, top):
    """The latest time on a (t, L) outline at or below distance `top`."""
    times = [t for t, l in polygon if l <= top]
    for (t0, l0), (t1, l1) in zip(polygon, polygon[1:] + polygon[:1]):
        if (l0 - top) * (l1 - top) < 0:
            times.append(t0 + (t1 - t0) * (top - l0) / (l1 - l0))
    return max(times, default=0.0)


def _log_intensity(ax, values):
    """A log intensity axis from 1e-4 of the peak of `values` to just above it."""
    peak = values.max(initial=0.0)
    if peak > 0:
        ax.set_yscale('log')
        ax.set_ylim(peak * 1e-4, peak * 2)


def draw_time_distance(result: Transmission, ax, top: float | None = None):
    """Where the transmitted neutrons are, and where the rest stop.

    Each band is what got through every chopper before one, from the source to that
    chopper. The bands are translucent, so where they overlap shows; the part of a band
    the next one does not cover is what that chopper stops. The last band, to the
    sample, is the outlined region. Each chopper is a line broken where it is open.
    `top` (m) shows only the instrument up to there.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    stopped = plt.get_cmap('YlGnBu')(np.linspace(0.25, 0.85, len(result.stages) - 1))
    colours = [*stopped, 'crimson']
    for (name, _), polygons, colour in zip(result.positions, result.stages, colours):
        label = 'reaches the sample' if name == 'sample' else f'stopped by {name}'
        for polygon in polygons:
            ax.add_patch(Polygon([(t * 1e3, l) for t, l in polygon], closed=True,
                                 facecolor=colour, edgecolor=colour, alpha=0.35,
                                 linewidth=0.8, label=label))
            label = None
    top = result.positions[-1][1] if top is None else top
    latest = max((_latest_below(p, top) for polygons in result.stages for p in polygons),
                 default=1 / 14)
    # each disc closed between its openings; a pair's lines lie on top of each other, so
    # its gaps show only where both discs are open
    end = 1.03 * latest
    for (_, at), windows in zip(result.positions, result.openings):
        if at > top:
            continue
        starts = [0.0] + [b for _, b in windows]
        ends = [a for a, _ in windows] + [end]
        closed = [(a * 1e3, b * 1e3) for a, b in zip(starts, ends) if a < b and a < end]
        ax.hlines([at] * len(closed), *zip(*closed), color='0.2', linewidth=1.4)
    # one name per housing: the pairs are centimetres apart
    housings = []
    for name, at in result.positions:
        if at > top:
            break
        if housings and at - housings[-1][1] < 0.04 * top:   # labels would overlap
            housings[-1][0].append(name)
        else:
            housings.append(([name], at))
    for names, at in housings:
        ax.annotate('/'.join(names), (1.0, at), xycoords=('axes fraction', 'data'),
                    xytext=(4, 0), textcoords='offset points', va='center',
                    fontsize='small')
    ax.axhline(result.positions[-1][1], color='0.4', linewidth=0.6)   # the sample
    ax.set_xlim(0.0, end * 1e3)
    ax.set_ylim(0.0, 1.03 * top)
    ax.set_xlabel('time [ms]')
    ax.set_ylabel('distance from the moderator [m]')
    if top < result.positions[-1][1]:
        ax.set_title('... and near the source', fontsize='medium')
    else:
        ax.set_title('time-distance: where the train stops each part of the source',
                     fontsize='medium')
        ax.legend(loc='upper left', fontsize='small')


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
        'latest_emission': FloatSlider(value=3.0, min=0.5, max=12.0, step=0.1,
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
        'log': Checkbox(value=False, description='log intensities'),
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


if __name__ == '__main__':
    # `swept` is exact: at every distance its band spans the times its vertices reach
    square = [(1e-3, 2e-3), (3e-3, 0.0), (3e-3, 1e-3), (2e-3, 3e-3)]   # bends at 1 m, 2 m
    band = np.array(swept(square, 0.0, 20.0))
    latest, earliest = band[:len(band) // 2], band[len(band) // 2:][::-1]
    for distance in np.linspace(0.0, 20.0, 41):
        reached = [t + distance * u for u, t in square]
        assert np.isclose(np.interp(distance, latest[:, 1], latest[:, 0]), max(reached))
        assert np.isclose(np.interp(distance, earliest[:, 1], earliest[:, 0]), min(reached))
    # the stages end with exactly what the train transmits, the outlined region
    result = transmission()
    assert len(result.stages[-1]) == len(result.outline)
    last = np.array(result.stages[-1][0])
    assert np.isclose(last[:, 0].min(), min(t for t, _ in result.outline[0]))
    # what the train transmits crosses every disc inside one of its openings
    from chopcal.lib import wavelength_to_inverse_velocity
    per_angstrom = wavelength_to_inverse_velocity(1.0)
    for (_, at), windows in zip(result.positions, result.openings):
        for polygon in result.outline:
            times = [t + (at - SOURCE_PLANE) * w * per_angstrom for t, w in polygon]
            assert any(a - 1e-9 <= min(times) and max(times) <= b + 1e-9
                       for a, b in windows), at
    print('ok')
