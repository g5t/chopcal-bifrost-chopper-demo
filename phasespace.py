"""An instrument's source through its chopper train, from chopcal's exact regions.

The source is a `Source`: an absolute rate, n/s, into the guide's entrance, binned on a
fine grid of (wavelength, emission time). `ESS` is McStas's ESS_butterfly for a beamline,
from `ess_butterfly`, pulsed at 14 Hz; `Continuous` is a steady source -- a reactor's, say
-- the same at every emission time, where the choppers make the pulses. chopcal builds the region of that plane the choppers transmit -- exactly,
as convex polygons -- and weights it by how likely each neutron is to get through a disc
whose edge is still sweeping the beam. Integrating the gridded source over that weighted
region gives what reaches the sample, by arrival time and by wavelength, and the mean and
spread of wavelength in each arrival-time bin.

What is particular to an instrument -- its source, its discs and how they are set, its
guide -- is an `Instrument`; `bifrost`, `trex` and `in5` define three. Three layers, so the same
numbers serve a notebook and a static picture:

- `transmission` computes everything for one instrument and setting, with no plotting;
- `draw` puts a result on `make_axes`'s axes;
- `figure` wires both to ipywidgets controls.

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

#: The colour of what reaches the sample through a region the train is not designed for.
SPURIOUS = 'C4'


class Source:
    """Where an instrument's neutrons come from: what `transmission` grids, and the
    controls `figure` gives it. Its settings are keyword arguments to `intensity` and
    `intended`, which `controls` makes and `settings` reads."""
    #: the source plane's time axis
    time_label = 'emission time at the source'
    #: the same at every time, so that only the train sets where the window starts
    steady = False

    def intensity(self, wavelength_edges, time_edges, solid_angle, **settings):
        """The time-averaged rate into each (emission time, wavelength) bin, n/s, through
        `solid_angle` (sr); shaped (time, wavelength)."""
        raise NotImplementedError

    def intended(self, polygons, **settings) -> bool:
        """Whether a transmitted region, as convex polygons of (inverse velocity, time),
        is one the train is designed to pass; the rest are drawn as spurious."""
        return True

    def controls(self) -> dict:
        """ipywidgets for the settings, by name."""
        return {}

    def settings(self, values: dict) -> dict:
        """`intensity`'s keyword arguments from the controls' values."""
        return values

    def note(self, values: dict) -> str:
        """A line shown under the source's controls."""
        return ''


class ESS(Source):
    """McStas's ESS_butterfly for one beamline, pulsed at 14 Hz. The settings are the
    proton pulse's duration (s), the beam power at the design pulse length (MW) -- the
    proton current stays the same, so a shorter pulse delivers proportionally fewer
    protons, and neutrons, per pulse -- and a factor on the moderator's decay times. What
    is emitted during the proton pulse is intended; what only after it, spurious."""

    def __init__(self, beamline: ess_butterfly.Beamline):
        self.beamline = beamline

    def intensity(self, wavelength_edges, time_edges, solid_angle,
                  power=ess_butterfly.REFERENCE_POWER,
                  pulse_duration=ess_butterfly.PULSE_DURATION, decay_scale=1.0):
        return self.beamline.intensity(
            wavelength_edges, time_edges, solid_angle,
            power=power * pulse_duration / ess_butterfly.PULSE_DURATION,
            duration=pulse_duration, decay_scale=decay_scale)

    def intended(self, polygons, pulse_duration=ess_butterfly.PULSE_DURATION, **_):
        return min(t for p in polygons for _, t in p) < pulse_duration

    def controls(self):
        from ipywidgets import FloatSlider
        return {
            'pulse_duration': FloatSlider(value=ess_butterfly.PULSE_DURATION * 1e3, min=0.1,
                                          max=3.0, step=0.01, readout_format='.3f',
                                          description='proton pulse [ms]'),
            'power': FloatSlider(value=ess_butterfly.REFERENCE_POWER, min=0.1, max=5.0,
                                 step=0.1, description='power at full pulse [MW]'),
            'decay_scale': FloatSlider(value=1.0, min=0.25, max=4.0, step=0.05,
                                       description='decay time ×'),
        }

    def settings(self, values):
        return {**values, 'pulse_duration': values['pulse_duration'] * 1e-3}

    def note(self, values):
        delivered = (values['power'] * values['pulse_duration'] * 1e-3
                     / ess_butterfly.PULSE_DURATION)
        return f'beam power {delivered:.2f} MW with this pulse'


class Continuous(Source):
    """A steady source: `spectrum(wavelength)` n/s per Å per sr, the same at every
    emission time. The time window is a stretch of a steady beam, so each bin holds the
    rate times its share of the window: what passes the train is the time-averaged rate
    exactly when the window spans whole periods of the train. Every region is intended;
    an instrument can say otherwise (`Instrument.intended`)."""
    time_label = 'time at the source'
    steady = True

    def __init__(self, spectrum, points=4):
        self.spectrum, self.points = spectrum, points

    def intensity(self, wavelength_edges, time_edges, solid_angle, scale=1.0):
        w = np.asarray(wavelength_edges, dtype=float)
        t = np.asarray(time_edges, dtype=float)
        nodes, weights = np.polynomial.legendre.leggauss(self.points)
        half = 0.5 * np.diff(w)
        at = (0.5 * (w[1:] + w[:-1]))[:, None] + half[:, None] * nodes
        per_bin = (self.spectrum(at) * weights).sum(axis=1) * half * solid_angle * scale
        return np.outer(np.diff(t) / (t[-1] - t[0]), per_bin)

    def controls(self):
        from ipywidgets import FloatLogSlider
        return {'scale': FloatLogSlider(value=1.0, min=-2, max=1, step=0.05,
                                        readout_format='.3g', description='flux ×')}


class Instrument:
    """What `transmission` and `figure` need to know of an instrument.

    A subclass sets the attributes and says how its discs are set, from settings of its
    own -- keyword arguments to `choppers`, which the controls `controls` makes give
    through `settings`.
    """
    name: str
    source: Source
    solid_angle: float                  # sr, the guide's entrance seen from the source
    sample: float                       # m from the source
    names: tuple                        # one per disc, upstream first
    groups: tuple                       # discs sharing a housing share a number
    delay_ranges_ms: dict               # slider ranges, by disc name
    #: what reaches the sample over what enters the guide, a CSV of wavelength (Å),
    #: fraction and its error; None for a perfect guide. An instrument whose guide
    #: changes with its settings says which table by `guide`.
    guide_table: Path | None = None
    wavelength_range: tuple = (1.0, 6.0)   # Å, the figure's starting source rectangle
    latest_emission: float = 6e-3          # s, the same
    wavelength_bin: float = 0.02           # Å, the source grid
    time_bin: float = 20e-6                # s, the same
    #: s, the arrival-time grid at the sample: what the moments of wavelength are per bin,
    #: so it must be finer than a band takes to arrive
    arrival_bin: float = 200e-6

    def choppers(self, **settings) -> list:
        """(disc, centre aperture) for every disc, in `names` order: a chopcal `Chopper`
        with the aperture of the beam's inner corner, which bounds what gets through,
        and the beam centre's, in degrees, which sets the ramps a weighted transmission
        models."""
        raise NotImplementedError

    def controls(self, **initial) -> dict:
        """ipywidgets for the settings, by name; changing one resets the delays."""
        raise NotImplementedError

    def settings(self, values: dict) -> dict:
        """`choppers`' keyword arguments from the controls' values."""
        return values

    def train_period(self, **settings) -> float | None:
        """How often the train repeats, s, for a steady source: `transmission` then
        takes the window in whole periods, starting between two transmitted regions."""
        return None

    def zoom(self, **settings) -> tuple | None:
        """The stretch of the instrument the second time-distance panel shows, (from, to)
        in m, and optionally the times it shows, (from, to) in s after them; None for the
        choppers near the source."""
        return None

    def note(self, result: 'Transmission', values: dict) -> str:
        """A line shown under the instrument's controls."""
        return ''

    def guide(self, **settings) -> Path | None:
        """The guide's table for these settings."""
        return self.guide_table

    def intended(self, polygons, settings: dict, source: dict) -> bool:
        """Whether a transmitted region -- convex polygons of (inverse velocity, time) --
        is one the train is designed for: the source's say, unless the instrument's."""
        return self.source.intended(polygons, **source)

    def guide_transmission(self, wavelength, /, **settings):
        """The fraction of what enters the guide that reaches the sample; interpolated
        in the settings' table, and held at its end values beyond it."""
        path = self.guide(**settings)
        if path is None:
            return np.ones_like(np.asarray(wavelength, dtype=float))
        table, value = _guide_table(path)
        return np.interp(wavelength, table, value)


@lru_cache(maxsize=4)
def _guide_table(path):
    return np.loadtxt(path, delimiter=',', skiprows=1, usecols=(0, 1)).T


def trains(instrument: Instrument, settings: dict, pulse_delay: float = 0.0,
           delays: dict | None = None):
    """The instrument's discs twice over: with the beam's inner-corner aperture, which
    bounds where anything gets through, and with its centre aperture, which sets the ramps
    a weighted transmission models.

    `pulse_delay` (s) is added to every disc's delay, as ESS's ``BeamPosDly-S`` is when
    the experiment delays NICOS sets stay as they are: the whole train moves together.
    `delays` (s), by disc name, replace the settings' delays for those discs; their speeds
    stay as set.
    """
    from chopcal.lib import Chopper
    delays = delays or {}
    support, ramped = [], []
    for name, (disc, centre) in zip(instrument.names, instrument.choppers(**settings)):
        delay = delays.get(name, disc.delay) + pulse_delay
        for train, aperture in ((support, disc.aperture), (ramped, centre)):
            train.append(Chopper(disc.speed, delay, disc.beam, list(disc.edges), disc.path,
                                 aperture))
    return support, ramped


@dataclass
class Transmission:
    """Everything one setting produces, in plain arrays."""
    wavelength_edges: np.ndarray        # Å
    time_edges: np.ndarray              # s, emission at the source
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
    speeds: list                        # Hz, each disc's, signed; 0 when parked
    #: (name, distance from the source in m) for each chopper, then the sample
    positions: list
    #: for each entry of `positions`, the (time, distance) outlines of the neutrons that
    #: got through every chopper before it, swept from the source up to it
    stages: list
    #: for each chopper, the (start, end) times in s that any of the beam can pass it,
    #: from 0 to past the latest time in `stages`
    openings: list
    #: the guide's transmission at each wavelength bin's centre
    guide: np.ndarray
    at_sample_guide: np.ndarray         # `at_sample` through the real guide
    transmitted_guide: float            # `transmitted` through the real guide
    #: for each polygon of `outline`, whether it is part of an intended region -- for
    #: ESS one emitted during the proton pulse -- or of a spurious one
    main: list
    #: the parts of `at_sample`, `by_wavelength`, `transmitted` and their guided
    #: versions that come through the spurious regions
    at_sample_spurious: np.ndarray
    by_wavelength_spurious: np.ndarray
    transmitted_spurious: float
    at_sample_guide_spurious: np.ndarray
    transmitted_guide_spurious: float
    zoom: tuple | None = None           # the instrument's, for `draw`
    time_label: str = 'emission time at the source'   # the source's
    #: s, what `clean_start` moved every disc by, for a steady source
    source_delay: float = 0.0


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


def transmission(instrument: Instrument, settings: dict | None = None,
                 wavelength_range=None, latest_emission=None, wavelength_bin=None,
                 time_bin=None, arrival_bin=None, pulse_delay=0.0, delays=None,
                 parked=(), **source) -> Transmission:
    """The source in a (wavelength, emission time) rectangle, through an instrument's train.

    `settings` are the keyword arguments of the instrument's `choppers`; its defaults
    without them. The rectangle defaults to the instrument's own.

    The rectangle is the source polygon chopcal transmits and the extent of the gridded
    source; the bin sizes set the grid. The source is the rate into the guide's entrance,
    n/s, from the instrument's `Source`, with `source` its settings -- for `ESS`, `power`,
    `pulse_duration` and `decay_scale`. `pulse_delay` moves every disc together; see
    `trains`. With a steady source and an instrument that gives its `train_period`, the
    window is cut to whole periods and every disc is moved together by `source_delay`,
    as `pulse_delay` moves them, so that the window starts between two transmitted
    regions (`clean_start`): it then holds only whole ones, and their time-averaged rate.
    Everything reaching the sample is for a perfect guide; `at_sample_guide` and
    `transmitted_guide` weight the source by the instrument's guide as well. The discs
    named in `parked` are stopped open: they pass everything, as if they were not there.
    Times in s, wavelengths in Å.
    """
    from chopcal.lib import Region, wavelength_to_inverse_velocity

    names = instrument.names
    low, high = sorted(wavelength_range or instrument.wavelength_range)
    latest_emission = latest_emission or instrument.latest_emission
    wavelength_bin = wavelength_bin or instrument.wavelength_bin
    time_bin = time_bin or instrument.time_bin
    arrival_bin = arrival_bin or instrument.arrival_bin
    wavelength_edges = np.arange(low, high + 0.5 * wavelength_bin, wavelength_bin)
    period = (instrument.train_period(**(settings or {}))
              if instrument.source.steady else None)
    if period:
        latest_emission = period * max(1, floor(latest_emission / period + 1e-9))
        time_edges = np.linspace(0.0, latest_emission,
                                 max(1, round(latest_emission / time_bin)) + 1)
    else:
        time_edges = np.arange(0.0, latest_emission + 0.5 * time_bin, time_bin)
    amounts = instrument.source.intensity(wavelength_edges, time_edges,
                                          instrument.solid_angle, **source)
    # ponytail: at bin centres, fine while the table is smooth on the bin size
    guide = instrument.guide_transmission(0.5 * (wavelength_edges[1:]
                                                 + wavelength_edges[:-1]),
                                          **(settings or {}))
    guided = amounts * guide

    per_angstrom = wavelength_to_inverse_velocity(1.0)   # chopper-lib's own conversion
    inverse_velocities = wavelength_edges * per_angstrom
    rectangle = Region.rectangle(inverse_velocities[0],
                                 inverse_velocities[-1] - inverse_velocities[0],
                                 time_edges[0], time_edges[-1] - time_edges[0])
    source_delay = (clean_start(instrument, settings or {}, period, inverse_velocities,
                                time_bin, pulse_delay, delays, parked) if period else 0.0)
    support_train, ramped_train = trains(instrument, settings or {},
                                         pulse_delay + source_delay, delays)
    running = [i for i, n in enumerate(names) if n not in parked]
    support = rectangle.transmit([support_train[i] for i in running])
    weighted = rectangle.transmit_weighted([ramped_train[i] for i in running],
                                           [instrument.groups[i] for i in running])

    distance = instrument.sample
    if support:
        reached = [t + distance * u for p in support.polygons for u, t in p.vertices]
        first, last = min(reached), max(reached)
        margin = 0.02 * (last - first)
        first, last = first - margin, last + margin
    else:
        first, last = time_edges[0], time_edges[-1]
    arrival_edges = first + arrival_bin * np.arange(ceil((last - first) / arrival_bin) + 1)

    def moments(region):
        return region.arrival_moments(amounts, inverse_velocities, time_edges, distance,
                                      arrival_edges)

    def by_wavelength(region):
        return (amounts * region.coverage(inverse_velocities, time_edges)).sum(axis=0)

    outline = [[(t, u / per_angstrom) for u, t in p.vertices] for p in support.polygons]

    # what gets through the first k choppers, swept from the moderator to the next one
    stops = [(f'{n} (parked)' if n in parked else n, d.path)
             for n, d in zip(names, support_train)] + [('sample', distance)]
    stages, through = [], rectangle
    for k, (_, stop) in enumerate(stops):
        if k and names[k - 1] not in parked:
            through = through.transmit([support_train[k - 1]])
        stages.append([swept(p.vertices, 0.0, stop) for p in through.polygons])
    positions = stops
    until = 1.1 * max((t for polygons in stages for q in polygons for t, _ in q),
                      default=time_edges[-1])
    openings = [[(0.0, until)] if n in parked else open_windows(chopper, until)
                for n, chopper in zip(names, support_train)]
    at_sample, at_sample_support = moments(weighted), moments(support)

    # the intended regions and the spurious ones, as the instrument or its source says:
    # every cell the train opens goes to the region nearest it -- inside one, its own
    # boundary is nearer than any other's
    # ponytail: a cell straddling two regions goes whole to one; they are far apart
    vertices = [p.vertices for p in support.polygons]
    groups = regions(vertices, (np.diff(inverse_velocities[:2])[0], time_bin))
    intended = [instrument.intended([vertices[i] for i in g], settings or {}, source)
                for g in groups]
    spurious_cells = np.zeros(amounts.shape, dtype=bool)
    main = [intended[k] for i in range(len(vertices))
            for k, g in enumerate(groups) if i in g]
    if not all(intended):
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
        spurious_cells = _cells(open_cells[~np.array(intended)[nearest]], amounts)
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
        emitted=float(amounts.sum()), guide=guide,
        speeds=[0.0 if n in parked else d.speed for n, d in zip(names, support_train)],
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
                                                      time_edges),
        zoom=instrument.zoom(**(settings or {})),
        time_label=instrument.source.time_label, source_delay=source_delay)


def clean_start(instrument: Instrument, settings: dict, period: float, inverse_velocities,
                time_bin: float, pulse_delay=0.0, delays=None, parked=()) -> float:
    """What to add to every disc's delay so that a window from 0 in whole periods holds
    only whole transmitted regions, with a steady source: the train repeats every
    `period`, and so do its regions, so a cut between two of them at one end is one at
    the other too.

    Two periods are transmitted, each region taken as the times it spans over the
    wavelengths, and those folded into one period; the middle of the widest gap between
    them moves to 0. With no gap -- a region a period long or more, or regions covering
    every time -- there is no clean start, and nothing moves.
    """
    from chopcal.lib import Region
    support, _ = trains(instrument, settings, pulse_delay, delays)
    running = [d for n, d in zip(instrument.names, support) if n not in parked]
    u = inverse_velocities
    through = Region.rectangle(u[0], u[-1] - u[0], 0.0, 2 * period).transmit(running)
    vertices = [p.vertices for p in through.polygons]
    if not vertices:
        return 0.0
    spans = []
    for group in regions(vertices, (u[1] - u[0], time_bin)):
        times = [t for i in group for _, t in vertices[i]]
        first, last = min(times), max(times)
        if last - first >= period:
            return 0.0
        spans.append((first % period, first % period + last - first))
    spans.sort()
    merged = [list(spans[0])]
    for a, b in spans[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    gaps = [(b[0] - a[1], a[1]) for a, b in zip(merged, merged[1:])]
    gaps.append((merged[0][0] + period - merged[-1][1], merged[-1][1]))   # round the end
    width, start = max(gaps)
    if width <= 0:
        return 0.0
    return -(start + width / 2) % period


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
    left.set_xlabel(f'{result.time_label} [ms]')
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
                linewidth=1.5, label='weighted, intended' if spurious
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
                      label='weighted, intended' if spurious else 'weighted by transmission')
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
        ratio = (f'{result.transmitted_spurious / main_total:.2g} of the intended'
                 if main_total else 'nothing intended')
        through += (f',\nspurious regions {result.transmitted_spurious:.3g} n/s, {ratio}')
    time.set_title(f'{text}{result.transmitted:.3g} n/s ({fraction:.3%} of the source '
                   f'rectangle) reach the sample{through}', fontsize='medium')
    # the whole instrument, then the choppers in its first quarter, where most is stopped
    # the whole instrument, then the instrument's zoom, or the choppers in its first
    # quarter, where most is stopped
    near = [at for _, at in result.positions if at < 0.25 * result.positions[-1][1]]
    zoom = result.zoom or (0.0, 1.15 * max(near, default=result.positions[-1][1]))
    for ax, window in zip(rest, (None, zoom)):
        draw_time_distance(result, ax, window)
    return colorbar


def _times_between(polygon, low, high):
    """The times a (t, L) outline reaches between distances `low` and `high`: its
    vertices there and where its edges cross either."""
    times = [t for t, l in polygon if low <= l <= high]
    for (t0, l0), (t1, l1) in zip(polygon, polygon[1:] + polygon[:1]):
        for at in (low, high):
            if (l0 - at) * (l1 - at) < 0:
                times.append(t0 + (t1 - t0) * (at - l0) / (l1 - l0))
    return times


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


def draw_time_distance(result: Transmission, ax, window: tuple | None = None):
    """Where the transmitted neutrons are, and where the rest stop.

    Each band is what got through every chopper before one, from the source to that
    chopper. The bands are translucent, so where they overlap shows; the part of a band
    the next one does not cover is what that chopper stops. The last band, to the
    sample, is the outlined region. Each chopper is a line broken where it is open.
    `window` (m), (from, to), shows only that stretch of the instrument, over the times
    the bands reach it -- or over (from, to) in s, if it gives those after the distances.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    stopped = plt.get_cmap('YlGnBu')(np.linspace(0.25, 0.85, len(result.stages) - 1))
    colours = [*stopped, 'crimson']
    for (name, _), polygons, colour in zip(result.positions, result.stages, colours):
        if name.endswith('(parked)'):   # stops nothing: its band is the next one's
            continue
        label = 'reaches the sample' if name == 'sample' else f'stopped by {name}'
        for polygon in polygons:
            ax.add_patch(Polygon([(t * 1e3, l) for t, l in polygon], closed=True,
                                 facecolor=colour, edgecolor=colour, alpha=0.35,
                                 linewidth=0.8, label=label))
            label = None
    sample = result.positions[-1][1]
    low, top, *span = window or (0.0, sample)
    if span:
        start, end = span
    else:
        times = [t for polygons in result.stages for p in polygons
                 for t in _times_between(p, low, top)] or [0.0, result.time_edges[-1]]
        margin = 0.03 * (max(times) - min(times))
        start = 0.0 if low == 0 else max(min(times) - margin, 0.0)
        end = max(times) + margin
    # each disc closed between its openings; a pair's lines lie on top of each other, so
    # its gaps show only where both discs are open
    for (_, at), windows in zip(result.positions, result.openings):
        if not low <= at <= top:
            continue
        starts = [0.0] + [b for _, b in windows]
        ends = [a for a, _ in windows] + [end]
        closed = [(a * 1e3, b * 1e3) for a, b in zip(starts, ends) if a < b and a < end]
        if closed:
            ax.hlines([at] * len(closed), *zip(*closed), color='0.2', linewidth=1.4)
        else:   # parked open
            ax.axhline(at, color='0.5', linestyle=':', linewidth=0.8)
    # one name per housing: the pairs are centimetres apart
    housings = []
    for name, at in result.positions:
        if not low <= at <= top:
            continue
        if housings and at - housings[-1][1] < 0.04 * (top - low):   # labels would overlap
            housings[-1][0].append(name)
        else:
            housings.append(([name], at))
    for names, at in housings:
        ax.annotate('/'.join(names), (1.0, at), xycoords=('axes fraction', 'data'),
                    xytext=(4, 0), textcoords='offset points', va='center',
                    fontsize='small')
    ax.axhline(result.positions[-1][1], color='0.4', linewidth=0.6)   # the sample
    ax.set_xlim(start * 1e3, end * 1e3)
    ax.set_ylim(low, top + 0.03 * (top - low))
    ax.set_xlabel('time [ms]')
    ax.set_ylabel('distance from the source [m]')
    if window:
        inside = [n for n, at in result.positions[:-1] if low <= at <= top]
        ax.set_title('... and near the source' if low == 0
                     else f"... and around {', '.join(inside)}", fontsize='medium')
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


def _figure(instrument: Instrument, **initial):
    from ipywidgets import (Checkbox, FloatLogSlider, FloatRangeSlider, FloatSlider, HBox,
                            HTML, Label, Layout, ToggleButton, VBox)
    import matplotlib.pyplot as plt

    style = {'description_width': '9em'}
    wide = Layout(width='26em')
    own = instrument.controls(**initial)   # what the delays follow
    source = instrument.source.controls()
    controls = {
        **own,
        'latest_emission': FloatSlider(value=instrument.latest_emission * 1e3, min=0.5,
                                       max=20.0, step=0.1, description='last emission [ms]'),
        'wavelength_range': FloatRangeSlider(value=instrument.wavelength_range, min=0.2,
                                             max=19.9, step=0.1,
                                             description='wavelengths [Å]'),
        **source,
        # the grids, on log scales: an instrument selecting narrow strips needs fine ones
        'time_bin': FloatLogSlider(value=instrument.time_bin * 1e6, min=0, max=np.log10(200),
                                   step=0.05, readout_format='.3g',
                                   description='time bin [µs]'),
        'wavelength_bin': FloatLogSlider(value=instrument.wavelength_bin, min=-3,
                                         max=np.log10(0.5), step=0.05, readout_format='.3g',
                                         description='wavelength bin [Å]'),
        'arrival_bin': FloatLogSlider(value=instrument.arrival_bin * 1e6, min=0, max=3,
                                      step=0.05, readout_format='.3g',
                                      description='arrival bin [µs]'),
        'pulse_delay': FloatSlider(value=0.0, min=-1000.0, max=1000.0, step=10.0,
                                   description='pulse delay [µs]'),
        'log': Checkbox(value=False, description='log intensities'),
    }
    if instrument.guide_table is not None:
        controls['guide'] = Checkbox(value=True, description='guide transmission')
    for widget in controls.values():
        widget.style, widget.layout, widget.continuous_update = style, wide, False

    # one delay per disc, following the instrument's settings, and free to move by hand
    # until one of those changes; the ranges widen if the settings ever go beyond them.
    # Each disc can be parked: stopped open.
    delays = {n: FloatSlider(value=0.0, min=lo, max=hi, step=0.01, readout_format='.2f',
                             description=f'{n} delay [ms]', continuous_update=False,
                             style=style, layout=Layout(width='22em'))
              for n, (lo, hi) in instrument.delay_ranges_ms.items()}
    parked = {n: ToggleButton(value=False, description='parked',
                              tooltip='stopped open: passes everything',
                              layout=Layout(width='6em'))
              for n in instrument.names}
    suggested = {n: Label() for n in instrument.names}   # what the settings say now
    state = {'colorbar': None, 'syncing': False}

    def settings():
        return instrument.settings({k: w.value for k, w in own.items()})

    def follow(*_):
        discs = [disc for disc, _ in instrument.choppers(**settings())]
        state['syncing'] = True   # one redraw for all of them, not one each
        try:
            for n, disc in zip(instrument.names, discs):
                value = disc.delay * 1e3
                suggested[n].value = f'set {value:.3f}'
                delays[n].min = min(delays[n].min, value)
                delays[n].max = max(delays[n].max, value)
                delays[n].value = value
        finally:
            state['syncing'] = False

    with plt.ioff():
        fig, axes = make_axes()
    status = Label()
    note = Label()      # the instrument's: what its settings come to
    power = Label()     # the source's: e.g. what the beam delivers with this pulse

    def update(*_):
        if state['syncing']:
            return
        v = {k: w.value for k, w in controls.items()}
        stopped = tuple(n for n in instrument.names if parked[n].value)
        for n in instrument.names:
            delays[n].disabled = n in stopped
        result = transmission(instrument, settings(),
                              wavelength_range=v['wavelength_range'],
                              latest_emission=v['latest_emission'] * 1e-3,
                              wavelength_bin=v['wavelength_bin'],
                              time_bin=v['time_bin'] * 1e-6,
                              arrival_bin=v['arrival_bin'] * 1e-6,
                              pulse_delay=v['pulse_delay'] * 1e-6,
                              delays={n: w.value * 1e-3 for n, w in delays.items()},
                              parked=stopped, **instrument.source.settings(
                                  {k: v[k] for k in source}))
        state['colorbar'] = draw(result, axes, colorbar=state['colorbar'], log=v['log'],
                                 guide=v.get('guide', False))
        fig.canvas.draw_idle()
        status.value = (f'{result.source.shape[1]} x {result.source.shape[0]} bins; '
                        f'{len(result.outline)} transmitted polygon(s)'
                        + (f'; every disc moved {result.source_delay * 1e3:.3f} ms to '
                           'start between regions' if result.source_delay else ''))
        note.value = instrument.note(result, {k: w.value for k, w in own.items()})
        power.value = instrument.source.note({k: v[k] for k in source})

    for widget in own.values():   # before `update`, so the delays are the settings' first
        widget.observe(follow, names='value')
    for widget in [*controls.values(), *delays.values(), *parked.values()]:
        widget.observe(update, names='value')
    follow()
    update()
    widgets = list(controls.values())
    widgets.insert(len(own), note)
    if source:
        widgets.insert(widgets.index(controls[list(source)[-1]]) + 1, power)
    rows = [HBox([delays[n], parked[n], suggested[n]]) for n in instrument.names]
    return HBox([fig.canvas, VBox([*widgets, status, HTML('<b>choppers</b>'), *rows])])


def figure(instrument: Instrument, **initial):
    """The interactive figure for an instrument: the source and the transmitted region on
    the left, what reaches the sample by arrival time and by wavelength on the right, and
    a time-distance diagram below. `initial` sets the instrument's own controls."""
    if ensure_widget_backend():
        return _figure(instrument, **initial)
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
    # touching pieces are one region, separate ones two
    a, b, c = [(0, 0), (1, 0), (1, 1)], [(1, 0), (2, 0), (1, 1)], [(5, 5), (6, 5), (6, 6)]
    assert sorted(map(sorted, regions([a, b, c], (1.0, 1.0)))) == [[0, 1], [2]]
    print('ok')
