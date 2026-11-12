# `chopcal` for BIFROST

An interactive notebook showing the ESS source BIFROST sees put through its six choppers,
computed exactly by [`chopcal`](https://pypi.org/project/chopcal/).

- **Left:** the source intensity over (emission time, wavelength), from
  [`tof`](https://tof.readthedocs.io)'s source library, with the region the chopper train
  transmits outlined.
- **Right:** what reaches the sample, by arrival time and by wavelength (energy along the
  top). Dashed curves count the whole transmitted region as open; solid ones weight it by
  how likely each neutron is to get through while a disc's edge is still sweeping the beam.
  On a second axis of the wavelength pane, in green, is the wavelength spread σλ of the
  neutrons arriving together, against their mean wavelength: how well arrival time pins
  wavelength down.
- **Below:** a time-distance diagram, showing which chopper stops each part of the source
  and what reaches the sample, over the whole instrument and near the source. Each
  chopper is a line with gaps where it is open.

The controls set the lowest energy and pulse-shaping time chopcal places the band at, the
source rectangle and the grid it is binned on, a pulse delay that moves every disc
together (as ESS's `BeamPosDly-S` does), and the fastest the pulse-shaping disks may turn,
as a harmonic of 14 Hz: the designed disks are rated to 210 Hz, the substitutes installed
now to 126 Hz. The speed chopcal picks, which a long burst can make slower, is shown under
that control.

The *guide transmission* toggle adds what reaches the sample through the real guide to
the arrival-time plot, on a second axis, and weights the source on the left by it. The
weight is the fraction of what enters the guide that reaches a 10 × 30 mm sample window,
from `guide_transmission.csv`: a McStas scan of niess's BIFROST with the choppers removed
(10⁶ rays per point, 0.5–12 Å). Outside that range it holds the end values.

Below those, each chopper's delay has its own slider. It follows chopcal whenever one of
chopcal's inputs (lowest energy, shaping time, PSC limit) changes, unless its lock is
pressed. A delay moved by hand stays put until one of those inputs changes.

A static version of this figure is on the chopcal documentation page
*See BIFROST's source through its train*.

## Setup

Everything comes from PyPI. With [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
git clone <this repository>
cd chopcal-bifrost-chopper-demo
uv sync                  # creates .venv from uv.lock
uv run jupyter lab bifrost_phasespace.ipynb
```

`uv sync` uses the versions pinned in `uv.lock`; `uv lock --upgrade` moves them to the
latest releases.

Without uv, any Python 3.11 or newer works:

```sh
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install "chopcal>=0.8" tof scipp numpy matplotlib ipympl ipywidgets jupyterlab
jupyter lab bifrost_phasespace.ipynb
```

The first run downloads the BIFROST source distribution (about 8 MB) into `tof`'s cache.

## Files

- `bifrost_phasespace.ipynb`: the notebook.
- `guide_transmission.csv`: the guide's transmission against wavelength.
- `polygonfigure.py`: the calculation and the plotting, in three layers usable on their
  own:
  - `transmission(...)` computes one setting and returns plain arrays, no plotting;
  - `make_axes()` and `draw(result, axes)` draw a result with matplotlib, for a script
    or a saved picture;
  - `draw_time_distance(result, ax)` draws only the time-distance diagram;
  - `bifrost_transmitted_phasespace_figure(...)` wires both to ipywidgets controls.

`python polygonfigure.py` runs its self-checks.

## Without Jupyter

```python
import matplotlib.pyplot as plt
from polygonfigure import draw, make_axes, transmission

result = transmission(energy_min=4.5, shaping_time=2e-4)   # meV, s
fig, axes = make_axes()
draw(result, axes)
fig.savefig('bifrost.png')
```
