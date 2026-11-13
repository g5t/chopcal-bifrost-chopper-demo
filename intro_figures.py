"""Diagrams for the introduction slides of `presentation.qmd`: what chopcal's parameters
describe, the plane it works in, and what the beam's width does.

    python intro_figures.py out_dir      # saves each as a PNG, to look at

The plane diagram is computed by chopcal itself, so its transmitted region is the real one.
"""
from __future__ import annotations

import functools

import numpy as np

H_OVER_M = 3956.034   # Å m/s
FONT = 17


def style():
    """The fonts these diagrams are drawn with, for a `with` block. Create any axes passed
    in inside it too; the rest of a notebook keeps its own."""
    import matplotlib
    return matplotlib.rc_context({'font.size': FONT, 'axes.titlesize': FONT + 1,
                                  'axes.labelsize': FONT})


def _styled(draw):
    @functools.wraps(draw)
    def styled(*args, **kwargs):
        with style():
            return draw(*args, **kwargs)
    return styled


@_styled
def disc(ax=None):
    """A disc seen along the beam, a moment before its zero mark reaches the beam, with
    every angle chopcal asks for drawn on it."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle, Wedge
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8), layout='constrained')
    # angles clockwise from the top, as a turning disc's edges reach the beam there
    at = lambda angle, r: (r * np.sin(np.radians(angle)), r * np.cos(np.radians(angle)))
    mark, radius = -55.0, 1.0           # where the zero mark is now; the beam is at 0
    edges = [(25, 50), (185, 210)]      # two openings, from the mark
    ax.add_patch(Circle((0, 0), radius, facecolor='0.85', edgecolor='0.3', linewidth=2))
    for a, b in edges:                  # theta1/theta2 are anticlockwise from +x
        ax.add_patch(Wedge((0, 0), radius * 0.97, 90 - (mark + b), 90 - (mark + a),
                           width=0.3, facecolor='white', edgecolor='0.3'))
    ax.add_patch(Circle((0, 0), 0.05, color='0.3'))
    # the beam where it crosses the disc, and its angular width there: the aperture
    ax.add_patch(Rectangle((-0.11, 0.70), 0.22, 0.24, facecolor='crimson', alpha=0.55,
                           edgecolor='crimson', linewidth=2))
    for side in (-1, 1):
        ax.plot([0, at(side * 9, 1.12)[0]], [0, at(side * 9, 1.12)[1]], color='crimson',
                linewidth=1, linestyle=':')
    ax.annotate('aperture: the beam\'s\nwidth on the disc', at(9, 1.13), xytext=(0.45, 1.32),
                color='crimson', arrowprops=dict(arrowstyle='-', color='crimson'))
    ax.text(0.14, 0.66, 'the beam\ncrosses here', color='crimson', ha='left', va='top')
    # the zero mark, and the angle from it to the beam
    ax.plot(*zip(at(mark, 0.85), at(mark, 1.08)), color='k', linewidth=3)
    ax.text(*at(mark, 1.22), 'zero\nmark', ha='right', va='center')
    ax.add_patch(Arc((0, 0), 2.7, 2.7, theta1=90, theta2=90 - mark, color='C0', linewidth=2))
    ax.text(-0.75, 1.6, 'beam: from the mark to\nwhere the beam crosses', color='C0',
            ha='center')
    # the edges, measured from the mark
    for a, r in zip(edges[0], (0.4, 0.55)):   # the first opening's two
        ax.add_patch(Arc((0, 0), 2 * r, 2 * r, theta1=90 - (mark + a), theta2=90 - mark,
                         color='C2', linewidth=2, linestyle='--'))
        ax.plot(*zip(at(mark + a, 0.1), at(mark + a, 0.7)), color='C2', linewidth=1)
    ax.plot(*zip(at(mark, 0.1), at(mark, 0.85)), color='k', linewidth=1, linestyle=':')
    ax.text(-0.2, -0.3, 'edges: each opening\'s two\nangles from the mark',
            color='C2', ha='center', va='center')
    # which way it turns: positive speed carries the mark towards the beam
    ax.add_patch(FancyArrowPatch(at(-150, 1.25), at(-110, 1.25), connectionstyle='arc3,rad=-0.2',
                                 arrowstyle='-|>', mutation_scale=25, color='k', linewidth=2))
    ax.text(*at(-140, 1.55), 'speed > 0', ha='center')
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.55, 1.75)
    ax.set_aspect('equal')
    ax.axis('off')
    return ax.figure


@_styled
def plane(axes=None):
    """Two discs, their openings as bands in (emission time, wavelength), and what passes
    both: on the left in time and distance, on the right in the source's plane."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from chopcal.lib import Chopper, Region, wavelength_to_inverse_velocity
    from phasespace import swept
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(16, 7), layout='constrained')
    left, right = axes
    per = wavelength_to_inverse_velocity(1.0)       # s/m per Å
    near, far = (6.0, 30.0)
    discs = [Chopper.centred(14.0, 2e-3 + near * 3.0 * per, 25.0, near),
             Chopper.centred(14.0, 2e-3 + far * 3.5 * per, 60.0, far)]
    t0, t1, w0, w1 = 0.0, 6e-3, 1.0, 6.0
    colours = ['C0', 'C1']
    # each opening: the emission times t with t + L u inside it, a band of slope -1/L
    w = np.array([w0, w1])
    for d, colour in zip(discs, colours):
        half = 0.5 * d.openings[0][1] / (360 * d.speed) - 0.5 * d.openings[0][0] / (360 * d.speed)
        lo, hi = d.delay - half, d.delay + half
        band = [(lo - d.path * w[0] * per, w[0]), (hi - d.path * w[0] * per, w[0]),
                (hi - d.path * w[1] * per, w[1]), (lo - d.path * w[1] * per, w[1])]
        right.add_patch(Polygon([(t * 1e3, l) for t, l in band], facecolor=colour, alpha=0.25,
                                edgecolor=colour, label=f'open at {d.path:g} m'))
    region = Region.rectangle(w0 * per, (w1 - w0) * per, t0, t1 - t0).transmit(discs)
    for p in region.polygons:
        right.add_patch(Polygon([(t * 1e3, u / per) for u, t in p.vertices], facecolor='none',
                                edgecolor='crimson', linewidth=3, label='passes both'))
        # and in time and distance: everything it holds, swept from the source on
        left.add_patch(Polygon([(t * 1e3, l) for t, l in swept(p.vertices, 0.0, far + 5)],
                               facecolor='crimson', alpha=0.35, edgecolor='crimson'))
    for d, colour in zip(discs, colours):
        half = (d.openings[0][1] - d.openings[0][0]) / (360 * d.speed) / 2
        for a, b in ((0, d.delay - half), (d.delay + half, 0.045)):
            left.hlines(d.path, a * 1e3, b * 1e3, color=colour, linewidth=5, zorder=3)
        left.text(44, d.path + 0.6, f'disc at {d.path:g} m', color=colour, ha='right')
    left.axhspan(-1, 0, color='0.8')
    left.text(0.3, -0.6, 'source', va='center')
    left.set_xlim(0, 45)
    left.set_ylim(-1, far + 5)
    left.set_xlabel('time [ms]')
    left.set_ylabel('distance [m]')
    left.set_title('time–distance:\na disc is a line, broken where it is open')
    right.set_xlim(t0 * 1e3, t1 * 1e3)
    right.set_ylim(w0, w1)
    right.set_xlabel('emission time [ms]')
    right.set_ylabel('wavelength [Å]')
    right.set_title('the source plane:\neach opening is a band')
    handles, labels = right.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    right.legend(unique.values(), unique.keys(), loc='upper right')
    return right.figure


@_styled
def ramp(ax=None):
    """What one disc passes against time: a point beam, and a beam with width."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 5.5), layout='constrained')
    opening, aperture = 4.0, 1.6        # in units of time at the disc
    t = np.linspace(-5, 9, 1401)
    point = (abs(t) <= opening / 2).astype(float)
    # the fraction of the beam an edge has uncovered rises across the aperture
    rise = np.clip((t + opening / 2 + aperture / 2) / aperture, 0, 1)
    fall = np.clip((opening / 2 + aperture / 2 - t) / aperture, 0, 1)
    wide = np.minimum(rise, fall)
    ax.plot(t, point, color='0.5', linestyle='--', linewidth=2, label='point beam')
    ax.fill_between(t, wide, color='C0', alpha=0.3)
    ax.plot(t, wide, color='C0', linewidth=3, label='real beam: chance to pass')
    edge = opening / 2 + aperture / 2
    ax.annotate('', (-edge, 1.12), (edge, 1.12), arrowprops=dict(arrowstyle='<->', color='crimson'))
    ax.text(0, 1.17, 'where anything passes: wider by the aperture', color='crimson', ha='center')
    ax.annotate('', (-edge, -0.12), (-edge + aperture, -0.12),
                arrowprops=dict(arrowstyle='<->', color='C0'))
    ax.text(-edge + aperture / 2, -0.24, 'aperture', color='C0', ha='center', va='top')
    ax.set_xlim(-5, 9)
    ax.set_ylim(-0.45, 1.35)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticks([])
    ax.set_xlabel('time at the disc')
    ax.set_ylabel('fraction passed')
    ax.legend(loc='center right', bbox_to_anchor=(1.0, 0.6), frameon=False)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    return ax.figure


if __name__ == '__main__':
    import sys
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    out = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
    for name, make in (('disc', disc), ('plane', plane), ('ramp', ramp)):
        make().savefig(out / f'intro_{name}.png', dpi=60)
    print('saved')
