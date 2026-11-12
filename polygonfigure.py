"""BIFROST's source intensity through its chopper train, from chopcal's exact regions.

The ESS source BIFROST sees is McStas's ESS_butterfly for its beamline, W4, from
`ess_butterfly`: an absolute rate, n/s, into the guide's entrance, binned on a fine grid of
(wavelength, emission time). chopcal builds the region of that plane the six choppers transmit --
exactly, as convex polygons -- and weights it by how likely each neutron is to get
through a disc whose edge is still sweeping the beam. Integrating the gridded source over
that weighted region gives what reaches the sample, by arrival time and by wavelength,
and the mean and spread of wavelength in each arrival-time bin.

Three layers, so the same numbers serve a notebook and a static picture:

- `transmission` computes everything for one setting, with no plotting;
- `draw` puts a result on three matplotlib axes;
- `bifrost_transmitted_phasespace_figure` wires both to ipywidgets controls.

Needs chopcal >= 0.8 (`Region.transmit_weighted`, `beam_aperture(..., at='centre')`,
`Region.arrival_moments`).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from math import ceil, floor
from pathlib import Path

import numpy as np

import ess_butterfly

#: BIFROST's beam port, and the solid angle its guide's entrance, 68.70 x 34.45 mm at
#: 1.9034 m from the moderator, takes from it (niess's BIFROST).
BEAMLINE = ess_butterfly.Beamline('W', 4)
GUIDE_SOLID_ANGLE = 0.068701 * 0.034454 / 1.90343 ** 2  # sr
#: The fastest the designed pulse-shaping disks may turn, Hz: the 15th harmonic of 14 Hz.
#: chopcal's own default is 196 Hz. The 126 Hz substitutes installed now are the 9th.
PSC_SPEED_MAX = 210.0
#: Delay slider ranges, ms: what chopcal.bifrost gives over the controls' ranges, and some.
DELAY_RANGES_MS = {'ps1': (-20.0, 20.0), 'ps2': (0.0, 40.0), 'fo1': (0.0, 25.0),
                   'fo2': (0.0, 40.0), 'bw1': (0.0, 180.0), 'bw2': (0.0, 180.0)}
#: The pulse-shaping pair and the bandwidth pair each share a housing.
GROUPS = [0, 0, 1, 2, 3, 3]
#: The colour of what reaches the sample through a region the train is not designed for.
SPURIOUS = 'C4'
NAMES = ('ps1', 'ps2', 'fo1', 'fo2', 'bw1', 'bw2')


#: BIFROST's guide transmission against wavelength, from McStas: what reaches a 10 x 30 mm
#: sample window over what enters the guide, with the choppers removed. Made by
#: bifrost-guide-transmission (niess's BIFROST, scanned with restage).
GUIDE_TRANSMISSION = Path(__file__).with_name('guide_transmission.csv')


@lru_cache(maxsize=1)
def _guide_table():
    return np.loadtxt(GUIDE_TRANSMISSION, delimiter=',', skiprows=1, usecols=(0, 1)).T


def guide_transmission(wavelength):
    """The fraction of what enters the guide that reaches a 10 x 30 mm sample window;
    interpolated in the table, and held at its end values beyond it."""
    table, value = _guide_table()
    return np.interp(wavelength, table, value)


def trains(energy_min: float, shaping_time: float,
           pulse_shaping_speed_max: float = PSC_SPEED_MAX,
           pulse_delay: float = 0.0, delays: dict | None = None):
    """BIFROST's discs twice over: once with the beam's inner-corner aperture, which bounds where anything
    gets through, and once with its centre aperture, which sets the ramps a weighted
    transmission models.

    `pulse_delay` (s) is added to every disc's delay, as ESS's ``BeamPosDly-S`` is when
    the experiment delays NICOS sets stay as they are: the whole train moves together.
    chopcal's settings are for none. `delays` (s), by chopper name, replace chopcal's
    delays for those choppers; their speeds stay chopcal's.
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

    delays = delays or {}

    def moved(name, aperture):
        disc = settings[name]
        return Chopper(disc.speed, delays.get(name, disc.delay) + pulse_delay, disc.beam,
                       list(disc.edges), disc.path, aperture)

    support = [moved(n, settings[n].aperture) for n in NAMES]
    ramped = [moved(n, centre['wide' if n.startswith('bw') else 'narrow']) for n in NAMES]
    return support, ramped


@dataclass
class Transmission:
    """Everything one setting produces, in plain arrays."""
    wavelength_edges: np.ndarray        # Å
    time_edges: np.ndarray              # s, emission at the moderator
    source: np.ndarray                  # (time, wavelength) n/s per bin into the guide
    outline: list                       # transmitted polygons as [(t, λ), ...]
    arrival_edges: np.ndarray           # s, at the sample
    at_sample: np.ndarray               # weighted amount per arrival bin
    at_sample_support: np.ndarray       # the same counting the whole region as open
    by_wavelength: np.ndarray           # weighted amount per wavelength bin
    #: per arrival bin, the mean wavelength and its standard deviation, Å; NaN where
    #: nothing arrives -- what a time-of-flight measurement can know of a neutron's
    #: wavelength from when it reaches the sample
    spread: tuple
    spread_support: tuple               # the same counting the whole region as open
    by_wavelength_support: np.ndarray
    transmitted: float                  # weighted total
    emitted: float                      # total in the source rectangle
    psc_speed: float                    # Hz, what the pulse-shaping pair turns at
    #: (name, distance from the moderator in m) for each chopper, then the sample
    positions: list
    #: for each entry of `positions`, the (time, distance) outlines of the neutrons that
    #: got through every chopper before it, swept from the moderator up to it
    stages: list
    #: for each chopper, the (start, end) times in s that any of the beam can pass it,
    #: from 0 to past the latest time in `stages`
    openings: list
    #: the guide's transmission at each wavelength bin's centre
    guide: np.ndarray
    at_sample_guide: np.ndarray         # `at_sample` through the real guide
    transmitted_guide: float            # `transmitted` through the real guide
    #: for each polygon of `outline`, whether it is part of the main region -- the one
    #: transmitting the most -- or of a spurious one
    main: list
    #: the parts of `at_sample`, `by_wavelength`, `transmitted` and their guided
    #: versions that come through the spurious regions
    at_sample_spurious: np.ndarray
    by_wavelength_spurious: np.ndarray
    transmitted_spurious: float
    at_sample_guide_spurious: np.ndarray
    transmitted_guide_spurious: float


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


def _segment_distance(points, vertices):
    """The distance from each of `points` (N, 2) to the boundary of a convex polygon."""
    a = np.asarray(vertices, dtype=float)
    b = np.roll(a, -1, axis=0)
    d = b - a
    length = np.maximum((d * d).sum(axis=1), 1e-300)
    rel = points[:, None, :] - a[None]                                    # (N, M, 2)
    f = np.clip((rel * d[None]).sum(axis=2) / length, 0.0, 1.0)
    return np.hypot(*np.moveaxis(rel - f[..., None] * d[None], 2, 0)).min(axis=1)


def regions(polygons, scale):
    """The convex pieces grouped into connected regions, as lists of indices.

    chopcal returns a region as convex pieces that share edges; pieces of one region
    touch, distinct regions do not. `scale` (2,) makes the coordinates comparable.
    """
    shapes = [np.asarray(p, dtype=float) / scale for p in polygons]
    group = list(range(len(shapes)))

    def root(i):
        while group[i] != i:
            i = group[i]
        return i

    for i, j in combinations(range(len(shapes)), 2):
        gap = min(_segment_distance(shapes[i], shapes[j]).min(),
                  _segment_distance(shapes[j], shapes[i]).min())
        if gap < 1e-6:
            group[root(i)] = root(j)
    out = {}
    for i in range(len(shapes)):
        out.setdefault(root(i), []).append(i)
    return list(out.values())


def transmission(energy_min=4.5, shaping_time=2e-4, wavelength_range=(1.0, 6.0),
                 latest_emission=3e-3, wavelength_bin=0.02, time_bin=20e-6,
                 arrival_bins=400, pulse_shaping_speed_max=PSC_SPEED_MAX, pulse_delay=0.0,
                 delays=None, power=ess_butterfly.REFERENCE_POWER,
                 pulse_duration=ess_butterfly.PULSE_DURATION,
                 decay_scale=1.0) -> Transmission:
    """The source in a (wavelength, emission time) rectangle, through BIFROST's train.

    The rectangle is the source polygon chopcal transmits and the extent of the gridded
    source; the bin sizes set the grid. The source is the rate into the guide's entrance,
    n/s, from `ess_butterfly`, with the proton pulse lasting `pulse_duration` (s) and the
    moderator's decay times multiplied by `decay_scale`. `power` (MW) is the beam's at the
    design pulse length: the proton current stays the same, so a shorter pulse delivers
    proportionally fewer protons, and neutrons, per pulse. `pulse_delay` moves every disc together; see
    `trains`. Everything reaching the sample is for a perfect guide; `at_sample_guide` and
    `transmitted_guide` weight the source by `guide_transmission` as well. Times in s,
    wavelengths in Å.
    """
    from chopcal import constants as c
    from chopcal.lib import Region, wavelength_to_inverse_velocity

    low, high = sorted(wavelength_range)
    wavelength_edges = np.arange(low, high + 0.5 * wavelength_bin, wavelength_bin)
    time_edges = np.arange(0.0, latest_emission + 0.5 * time_bin, time_bin)
    amounts = BEAMLINE.intensity(wavelength_edges, time_edges, GUIDE_SOLID_ANGLE,
                                 power=power * pulse_duration / ess_butterfly.PULSE_DURATION,
                                 duration=pulse_duration, decay_scale=decay_scale)
    # ponytail: at bin centres, fine while the table is smooth on the bin size
    guide = guide_transmission(0.5 * (wavelength_edges[1:] + wavelength_edges[:-1]))
    guided = amounts * guide

    per_angstrom = wavelength_to_inverse_velocity(1.0)   # chopper-lib's own conversion
    inverse_velocities = wavelength_edges * per_angstrom
    rectangle = Region.rectangle(inverse_velocities[0],
                                 inverse_velocities[-1] - inverse_velocities[0],
                                 time_edges[0], time_edges[-1] - time_edges[0])
    support_train, ramped_train = trains(energy_min, shaping_time, pulse_shaping_speed_max,
                                         pulse_delay, delays)
    support = rectangle.transmit(support_train)
    weighted = rectangle.transmit_weighted(ramped_train, GROUPS)

    distance = c.INSTRUMENT_LENGTH
    if support:
        reached = [t + distance * u for p in support.polygons for u, t in p.vertices]
        first, last = min(reached), max(reached)
        margin = 0.02 * (last - first)
        arrival_edges = np.linspace(first - margin, last + margin, arrival_bins + 1)
    else:
        arrival_edges = np.linspace(0.0, 1 / 14, arrival_bins + 1)

    def moments(region):
        return region.arrival_moments(amounts, inverse_velocities, time_edges, distance,
                                      arrival_edges)

    def by_wavelength(region):
        return (amounts * region.coverage(inverse_velocities, time_edges)).sum(axis=0)

    outline = [[(t, u / per_angstrom) for u, t in p.vertices] for p in support.polygons]

    # what gets through the first k choppers, swept from the moderator to the next one
    stops = [(n, d.path) for n, d in zip(NAMES, support_train)] + [('sample', distance)]
    stages, through = [], rectangle
    for k, (_, stop) in enumerate(stops):
        if k:
            through = through.transmit([support_train[k - 1]])
        stages.append([swept(p.vertices, 0.0, stop) for p in through.polygons])
    positions = stops
    until = 1.1 * max((t for polygons in stages for q in polygons for t, _ in q),
                      default=1 / 14)
    openings = [open_windows(chopper, until) for chopper in support_train]
    at_sample, at_sample_support = moments(weighted), moments(support)

    # the main region and the spurious ones: every cell the train opens goes to the
    # region nearest it -- inside one, its own boundary is nearer than any other's
    # ponytail: a cell straddling two regions goes whole to one; they are far apart
    vertices = [p.vertices for p in support.polygons]
    groups = regions(vertices, (np.diff(inverse_velocities[:2])[0], time_bin))
    spurious_cells = np.zeros(amounts.shape, dtype=bool)
    main = [True] * len(vertices)
    if len(groups) > 1:
        open_cells = np.argwhere(support.coverage(inverse_velocities, time_edges) > 0)
        centres = np.column_stack([
            0.5 * (inverse_velocities[open_cells[:, 1]]
                   + inverse_velocities[open_cells[:, 1] + 1]),
            0.5 * (time_edges[open_cells[:, 0]] + time_edges[open_cells[:, 0] + 1])])
        scale = np.array([np.diff(inverse_velocities[:2])[0], time_bin])
        distances = np.array([
            np.min([_segment_distance(centres / scale, np.asarray(vertices[i]) / scale)
                    for i in g], axis=0) for g in groups])
        nearest = distances.argmin(axis=0)
        shares = [weighted.integrate(np.where(_cells(open_cells[nearest == k], amounts),
                                              amounts, 0.0), inverse_velocities, time_edges)
                  for k in range(len(groups))]
        first = int(np.argmax(shares))
        spurious_cells = _cells(open_cells[nearest != first], amounts)
        main = [i in groups[first] for i in range(len(vertices))]
    spurious = np.where(spurious_cells, amounts, 0.0)
    guided_spurious = spurious * guide
    return Transmission(
        wavelength_edges=wavelength_edges, time_edges=time_edges, source=amounts,
        outline=outline, arrival_edges=arrival_edges,
        at_sample=at_sample.amount, at_sample_support=at_sample_support.amount,
        spread=(at_sample.wavelength, at_sample.sigma),
        spread_support=(at_sample_support.wavelength, at_sample_support.sigma),
        by_wavelength=by_wavelength(weighted), by_wavelength_support=by_wavelength(support),
        transmitted=weighted.integrate(amounts, inverse_velocities, time_edges),
        emitted=float(amounts.sum()), psc_speed=abs(support_train[0].speed), guide=guide,
        at_sample_guide=weighted.arrivals(guided, inverse_velocities, time_edges, distance,
                                          arrival_edges).sum(axis=1),
        transmitted_guide=weighted.integrate(guided, inverse_velocities, time_edges),
        positions=positions, stages=stages, openings=openings, main=main,
        at_sample_spurious=weighted.arrivals(spurious, inverse_velocities, time_edges,
                                             distance, arrival_edges).sum(axis=1),
        by_wavelength_spurious=(spurious
                                * weighted.coverage(inverse_velocities, time_edges)).sum(axis=0),
        transmitted_spurious=weighted.integrate(spurious, inverse_velocities, time_edges),
        at_sample_guide_spurious=weighted.arrivals(
            guided_spurious, inverse_velocities, time_edges, distance,
            arrival_edges).sum(axis=1),
        transmitted_guide_spurious=weighted.integrate(guided_spurious, inverse_velocities,
                                                      time_edges))


def _cells(indices, like):
    """A boolean mask shaped `like`, True at the (time, wavelength) `indices`."""
    mask = np.zeros(like.shape, dtype=bool)
    mask[tuple(indices.T)] = True
    return mask


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
         log: bool = False, guide: bool = False):
    """Draw one result on `make_axes`'s axes, replacing what was there.

    `guide` shows the source through the real guide, what of it reaches the sample by
    arrival time on a second axis, and by wavelength in place of the perfect guide's.

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
    for ax in list(left.figure.axes):   # the twinned spread axis from the last draw
        if ax.get_label() in ('spread', 'guided'):
            ax.remove()

    w, t = result.wavelength_edges, result.time_edges
    source = result.source * result.guide if guide else result.source
    density = source / np.outer(np.diff(t * 1e3), np.diff(w))   # per ms per Å
    positive = density[density > 0]
    if not positive.size:
        norm = None
    elif log:
        norm = LogNorm(vmin=positive.max() * 1e-4, vmax=positive.max())
    else:
        norm = Normalize(vmin=0.0, vmax=positive.max())
    # nothing left at all (short wavelengths, long after the pulse) is the bottom colour
    import matplotlib
    cmap = matplotlib.colormaps['viridis']
    cmap = cmap.with_extremes(bad=cmap(0.0), under=cmap(0.0))
    mesh = left.pcolormesh(t * 1e3, w, density.T, norm=norm, cmap=cmap,
                           shading='flat', rasterized=True)
    for polygon, main in zip(result.outline, result.main):
        left.add_patch(Polygon([(a * 1e3, b) for a, b in polygon], closed=True,
                               fill=False, edgecolor='white', linewidth=1.6))
        left.add_patch(Polygon([(a * 1e3, b) for a, b in polygon], closed=True,
                               fill=False, edgecolor='crimson' if main else SPURIOUS,
                               linewidth=0.9))
    left.set_xlim(t[0] * 1e3, t[-1] * 1e3)
    left.set_ylim(w[0], w[-1])
    left.set_xlabel('emission time at the source [ms]')
    left.set_ylabel('wavelength [Å]')
    energy = left.secondary_yaxis('right', functions=(_energy, _wavelength))
    _energy_axis(energy.yaxis, (w[0], w[-1]))
    energy.set_ylabel('energy [meV]')
    left.set_title(('source through the guide' if guide else 'source into the guide')
                   + ', and the region the train transmits', fontsize='medium')
    if colorbar is None:
        colorbar = left.figure.colorbar(mesh, ax=left, location='bottom', shrink=0.8,
                                        label='source [n/s per ms per Å]')
    else:
        colorbar.update_normal(mesh)

    a = result.arrival_edges * 1e3
    step = np.diff(a)
    time.stairs(result.at_sample_support / step, a, color='0.6', linestyle='--',
                label='whole region open')
    spurious = result.transmitted_spurious > 0
    time.stairs((result.at_sample - result.at_sample_spurious) / step, a, color='C0',
                linewidth=1.5, label='weighted, main region' if spurious
                else 'weighted by transmission')
    if spurious:
        time.stairs(result.at_sample_spurious / step, a, color=SPURIOUS, linewidth=1.3,
                    label='weighted, spurious')
    time.set_xlabel('arrival time at the sample [ms]')
    time.set_ylabel('intensity [n/s per ms]')
    time.set_xlim(a[0], a[-1])
    handles = time.get_legend_handles_labels()[0]
    if guide:   # an order of magnitude below, so on its own axis
        guided = time.twinx()
        guided.set_label('guided')
        guided.stairs((result.at_sample_guide - result.at_sample_guide_spurious) / step, a,
                      color='C3', linewidth=1.3, label='through the guide')
        if spurious:
            guided.stairs(result.at_sample_guide_spurious / step, a, color='C3',
                          linestyle=':', linewidth=1.3, label='spurious, through the guide')
        guided.set_ylim(bottom=0.0)
        guided.set_ylabel('through the guide [n/s per ms]', color='C3')
        guided.tick_params(axis='y', colors='C3')
        handles += guided.get_legend_handles_labels()[0]
        if log:
            _log_intensity(guided, result.at_sample_guide / step,
                           result.at_sample_guide_spurious / step)
    time.legend(handles=handles, loc='upper left',
                bbox_to_anchor=(1.2 if guide else 1.01, 1.0), fontsize='small')
    if log:
        _log_intensity(time, result.at_sample_support / step,
                       result.at_sample_spurious / step)

    wstep = np.diff(w)
    # the guide weights each wavelength bin by its own transmission, so it multiplies these
    through = result.guide if guide else 1.0
    wavelength.stairs(result.by_wavelength_support * through / wstep, w, color='0.6',
                      linestyle='--', label='whole region open')
    wavelength.stairs((result.by_wavelength - result.by_wavelength_spurious) * through
                      / wstep, w, color='C1', linewidth=1.5,
                      label='weighted, main region' if spurious else 'weighted by transmission')
    if spurious:
        wavelength.stairs(result.by_wavelength_spurious * through / wstep, w,
                          color=SPURIOUS, linewidth=1.3, label='weighted, spurious')
    reached = np.nonzero(result.by_wavelength_support)[0]
    if reached.size:
        lo, hi = w[max(reached[0] - 2, 0)], w[min(reached[-1] + 3, len(w) - 1)]
        wavelength.set_xlim(lo, hi)
    wavelength.set_xlabel('wavelength [Å]')
    wavelength.set_ylabel(('through the guide ' if guide else 'intensity ')
                          + '[n/s per Å]')
    top = wavelength.secondary_xaxis('top', functions=(_energy, _wavelength))
    _energy_axis(top.xaxis, wavelength.get_xlim())
    top.set_xlabel('energy [meV]')
    if log:
        _log_intensity(wavelength, result.by_wavelength_support * through / wstep,
                       result.by_wavelength_spurious * through / wstep)
    # the wavelength spread of what arrives together, against its mean wavelength
    spread = wavelength.twinx()
    spread.set_label('spread')
    for (mean, sigma), style, label in (
            (result.spread_support, dict(linestyle='--', linewidth=0.9), 'σλ, whole open'),
            (result.spread, dict(linewidth=1.3), 'σλ, weighted')):
        spread.plot(mean, sigma * 1e3, color='C2', label=label, **style)
    spread.set_ylim(bottom=0.0)
    spread.set_ylabel('σλ per arrival time [mÅ]', color='C2')
    spread.tick_params(axis='y', colors='C2')
    handles = [h for ax in (wavelength, spread) for h in ax.get_legend_handles_labels()[0]]
    wavelength.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.12, 1.0),
                      fontsize='small')

    fraction = result.transmitted / result.emitted if result.emitted else 0.0
    text = title + '\n' if title else ''
    through = (f',\n{result.transmitted_guide:.3g} n/s '
               f'({result.transmitted_guide / result.emitted:.3%}) through the guide'
               if guide and result.emitted else '')
    if spurious:
        main_total = result.transmitted - result.transmitted_spurious
        through += (f',\nspurious regions {result.transmitted_spurious:.3g} n/s, '
                    f'{result.transmitted_spurious / main_total:.2g} of the main')
    time.set_title(f'{text}{result.transmitted:.3g} n/s ({fraction:.3%} of the source '
                   f'rectangle) reach the sample{through}', fontsize='medium')
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


def _log_intensity(ax, values, spurious=None):
    """A log intensity axis from 1e-4 of the peak of `values` to just above it, or
    lower, to a tenth of the peak of `spurious`, if that is lower still."""
    peak = values.max(initial=0.0)
    if peak > 0:
        low = peak * 1e-4
        if spurious is not None and spurious.max(initial=0.0) > 0:
            low = min(low, 0.1 * spurious.max())
        ax.set_yscale('log')
        ax.set_ylim(low, peak * 2)


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
        # under the whole figure, where it covers nothing; replace the last draw's
        for legend in list(ax.figure.legends):
            legend.remove()
        ax.figure.legend(*ax.get_legend_handles_labels(), loc='outside lower center',
                         ncols=7, fontsize='small')


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
    from ipywidgets import (Checkbox, FloatRangeSlider, FloatSlider, HBox, HTML, Label,
                            Layout, SelectionSlider, ToggleButton, VBox)
    import chopcal
    import matplotlib.pyplot as plt

    style = {'description_width': '9em'}
    wide = Layout(width='26em')
    controls = {
        'energy_min': FloatSlider(value=ei_min, min=1.0, max=20.0, step=0.1,
                                  description='lowest energy [meV]'),
        'shaping_time': FloatSlider(value=t_psc, min=0.05, max=4.0, step=0.05,
                                    description='shaping time [ms]'),
        'latest_emission': FloatSlider(value=6.0, min=0.5, max=20.0, step=0.1,
                                       description='last emission [ms]'),
        # the source: the proton current stays the same, so a shorter pulse is less power
        'pulse_duration': FloatSlider(value=ess_butterfly.PULSE_DURATION * 1e3, min=0.1,
                                      max=3.0, step=0.01, readout_format='.3f',
                                      description='proton pulse [ms]'),
        'power': FloatSlider(value=ess_butterfly.REFERENCE_POWER, min=0.1, max=5.0,
                             step=0.1, description='power at full pulse [MW]'),
        'decay_scale': FloatSlider(value=1.0, min=0.25, max=4.0, step=0.05,
                                   description='decay time ×'),
        'wavelength_range': FloatRangeSlider(value=(1.0, 6.0), min=0.2, max=19.9, step=0.1,
                                             description='wavelengths [Å]'),
        'time_bin': FloatSlider(value=20.0, min=6.0, max=200.0, step=2.0,
                                description='time bin [µs]'),
        'wavelength_bin': FloatSlider(value=0.02, min=0.02, max=0.5, step=0.01,
                                      description='wavelength bin [Å]'),
        'pulse_delay': FloatSlider(value=0.0, min=-1000.0, max=1000.0, step=10.0,
                                   description='pulse delay [µs]'),
        # the disks turn at a harmonic of 14 Hz, so their limit is one too
        'psc_speed_max': SelectionSlider(options=[(f'{14 * n} Hz', 14.0 * n)
                                                  for n in range(1, 16)],
                                         value=PSC_SPEED_MAX, description='PSC limit'),
        'log': Checkbox(value=False, description='log intensities'),
        'guide': Checkbox(value=True, description='guide transmission'),
    }
    for widget in controls.values():
        widget.style, widget.layout, widget.continuous_update = style, wide, False


    # one delay per disc, following chopcal until frozen, and free to move by hand until
    # one of chopcal's own inputs changes; ranges cover chopcal over these controls' ranges
    # and widen if chopcal ever goes beyond them
    delays = {n: FloatSlider(value=0.0, min=lo, max=hi, step=0.01, readout_format='.2f',
                             description=f'{n} delay [ms]', continuous_update=False,
                             style=style, layout=Layout(width='22em'))
              for n, (lo, hi) in DELAY_RANGES_MS.items()}
    frozen = {n: ToggleButton(value=False, icon='lock', tooltip='keep this delay when '
                              "chopcal's inputs change", layout=Layout(width='3em'))
              for n in NAMES}
    suggested = {n: Label() for n in NAMES}   # what chopcal says now, frozen or not
    chopcal_inputs = ('energy_min', 'shaping_time', 'psc_speed_max')
    state = {'colorbar': None, 'syncing': False}

    def follow_chopcal(*_):
        settings = chopcal.bifrost(energy_min=controls['energy_min'].value,
                                   shaping_time=controls['shaping_time'].value * 1e-3,
                                   pulse_shaping_speed_max=controls['psc_speed_max'].value)
        state['syncing'] = True   # one redraw for all six, not one each
        try:
            for n in NAMES:
                value = settings[n].delay * 1e3
                suggested[n].value = f'chopcal {value:.3f}'
                if not frozen[n].value:
                    delays[n].min = min(delays[n].min, value)
                    delays[n].max = max(delays[n].max, value)
                    delays[n].value = value
        finally:
            state['syncing'] = False

    with plt.ioff():
        fig, axes = make_axes()
    status = Label()
    psc = Label()   # what chopcal picked: the fastest harmonic the burst allows
    power = Label()   # what the beam delivers with this pulse

    def update(*_):
        if state['syncing']:
            return
        v = {k: w.value for k, w in controls.items()}
        result = transmission(energy_min=v['energy_min'], shaping_time=v['shaping_time'] * 1e-3,
                              wavelength_range=v['wavelength_range'],
                              latest_emission=v['latest_emission'] * 1e-3,
                              wavelength_bin=v['wavelength_bin'],
                              time_bin=v['time_bin'] * 1e-6,
                              pulse_delay=v['pulse_delay'] * 1e-6,
                              pulse_shaping_speed_max=v['psc_speed_max'],
                              delays={n: w.value * 1e-3 for n, w in delays.items()},
                              power=v['power'], pulse_duration=v['pulse_duration'] * 1e-3,
                              decay_scale=v['decay_scale'])
        state['colorbar'] = draw(result, axes, colorbar=state['colorbar'], log=v['log'],
                                 guide=v['guide'])
        fig.canvas.draw_idle()
        status.value = (f'{result.source.shape[1]} x {result.source.shape[0]} bins; '
                        f'{len(result.outline)} transmitted polygon(s)')
        slower = (' -- slower than the limit, for a burst this long'
                  if result.psc_speed < v['psc_speed_max'] else '')
        psc.value = f'PSC turns at {result.psc_speed:g} Hz{slower}'
        delivered = v['power'] * v['pulse_duration'] * 1e-3 / ess_butterfly.PULSE_DURATION
        power.value = f'beam power {delivered:.2f} MW with this pulse'

    for name in chopcal_inputs:   # before `update`, so the delays are chopcal's first
        controls[name].observe(follow_chopcal, names='value')
    for widget in [*controls.values(), *delays.values()]:
        widget.observe(update, names='value')
    follow_chopcal()
    update()
    widgets = list(controls.values())
    widgets.insert(widgets.index(controls['psc_speed_max']) + 1, psc)
    widgets.insert(widgets.index(controls['power']) + 1, power)
    rows = [HBox([delays[n], frozen[n], suggested[n]]) for n in NAMES]
    return HBox([fig.canvas, VBox([*widgets, status, HTML('<b>chopper delays</b>'), *rows])])


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
            times = [t + at * w * per_angstrom for t, w in polygon]
            assert any(a - 1e-9 <= min(times) and max(times) <= b + 1e-9
                       for a, b in windows), at
    # the guide only removes neutrons, least at long wavelengths
    assert np.all(guide_transmission([0.5, 2.0, 12.0, 30.0]) <= 1.0)
    assert guide_transmission(1.0) < 0.005 < 0.05 < guide_transmission(10.0)
    assert result.transmitted_guide < 0.1 * result.transmitted
    # by wavelength, the guide multiplies each bin: the profile sums to the guided total
    assert np.isclose((result.by_wavelength * result.guide).sum(), result.transmitted_guide)
    # replacing every delay by chopcal's plus x moves the train as a pulse delay x does
    import chopcal
    settings = chopcal.bifrost(energy_min=4.5, shaping_time=2e-4,
                               pulse_shaping_speed_max=PSC_SPEED_MAX)
    moved = transmission(delays={n: settings[n].delay + 1e-4 for n in NAMES})
    assert np.allclose(moved.outline, transmission(pulse_delay=1e-4).outline)
    # a long emission window opens a second, spurious region, far weaker than the main;
    # its parts are within the totals, and the main region is the one designed
    late = transmission(latest_emission=15e-3, wavelength_range=(1.0, 8.0))
    assert late.main.count(False) >= 1 and result.main == [True] * len(result.outline)
    assert 0 < late.transmitted_spurious < 1e-6 * late.transmitted
    assert np.isclose(late.at_sample_spurious.sum(), late.transmitted_spurious)
    assert np.isclose(late.transmitted - late.transmitted_spurious, result.transmitted,
                      rtol=0.02)
    assert all(min(t for t, _ in late.outline[i]) > 5e-3
               for i, main in enumerate(late.main) if not main)
    # the source knobs: power scales everything; at the same current, half the pulse is
    # half the neutrons, once the window holds all of both pulses
    assert np.isclose(transmission(power=2.5).transmitted, 0.5 * result.transmitted)
    whole = transmission(latest_emission=20e-3).emitted
    half = transmission(latest_emission=20e-3,
                        pulse_duration=ess_butterfly.PULSE_DURATION / 2).emitted
    assert np.isclose(half, 0.5 * whole, rtol=1e-3), half / whole

    # the delay sliders follow chopcal until frozen, and keep a hand-set value until one
    # of chopcal's inputs changes
    import matplotlib
    matplotlib.use('module://ipympl.backend_nbagg')
    panel = _bifrost_transmitted_phasespace_figure().children[1].children
    control = {w.description: w for w in panel if hasattr(w, 'description')}
    rows = {r.children[0].description.split()[0]: r.children for r in panel
            if type(r).__name__ == 'HBox'}

    def chopcal_ms(name):
        return chopcal.bifrost(energy_min=control['lowest energy [meV]'].value,
                               shaping_time=control['shaping time [ms]'].value * 1e-3,
                               pulse_shaping_speed_max=control['PSC limit'].value
                               )[name].delay * 1e3

    assert all(np.isclose(rows[n][0].value, chopcal_ms(n)) for n in NAMES)
    rows['ps1'][0].value = 1.234                   # by hand
    rows['fo1'][1].value = True                    # frozen
    fo1 = rows['fo1'][0].value
    control['last emission [ms]'].value = 4.0      # not one of chopcal's inputs
    assert rows['ps1'][0].value == 1.234
    control['lowest energy [meV]'].value = 7.0     # one of chopcal's inputs
    assert np.isclose(rows['ps1'][0].value, chopcal_ms('ps1'))
    assert rows['fo1'][0].value == fo1 and not np.isclose(fo1, chopcal_ms('fo1'))
    print('ok')
