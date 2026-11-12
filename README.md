# `chopcal` for BIFROST and T-REX

Interactive notebooks showing the ESS source an instrument sees put through its choppers,
computed exactly by [`chopcal`](https://pypi.org/project/chopcal/): BIFROST in
`bifrost_phasespace.ipynb`, T-REX in `trex_phasespace.ipynb`. What follows describes
BIFROST's; T-REX's is the same figure, with its own settings (central wavelength L0,
repetition rate multiplication of 4 to 24 in steps of 4, HF or HR), a zoom on its M
choppers rather than on those near the source, and, so far, a perfect guide.

- **Left:** the source over (emission time, wavelength), in neutrons per second into the
  guide's entrance, from `ess_butterfly.py`, with the region the chopper train transmits
  outlined.
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
source rectangle and the grid it is binned on (with the arrival-time grid, whose bins
the wavelength moments are taken over), a pulse delay that moves every disc
together (as ESS's `BeamPosDly-S` does), and the fastest the pulse-shaping disks may turn,
as a harmonic of 14 Hz: the designed disks are rated to 210 Hz, the substitutes installed
now to 126 Hz. The speed chopcal picks, which a long burst can make slower, is shown under
that control.

Three controls change the source itself: the proton pulse's duration, the beam power at
the full 2.857 ms pulse, and a factor on the moderator's decay times. The proton current
is held fixed, so a shorter pulse delivers proportionally less power and fewer neutrons
per pulse; the power it gives is shown under the control.
The decay is fitted over the first few pulse lengths, so beyond about 9 ms it is an
extrapolation; the factor shows how much the long-emission region depends on it.

The *guide transmission* toggle, on by default, adds what reaches the sample through the
real guide to the arrival-time plot, on a second axis. It also makes the wavelength plot
show intensity through the guide, and weights the source on the left by it. The
weight is the fraction of what enters the guide that reaches a 10 × 30 mm sample window,
from `bifrost_guide_transmission.csv`: a McStas scan of niess's BIFROST with the choppers removed
(10⁶ rays per point, 0.5–12 Å). Outside that range it holds the end values.

If the train opens more than one region of the source, those emitted during the proton
pulse are intended, outlined in red; any emitted only after it are spurious, outlined in
purple. Both profiles then split what reaches the sample into the two, with the spurious
part in purple (dotted red through the guide), and the title gives its total and its
fraction of the intended. It is usually many orders of magnitude down, so use *log intensities* to see
it; the log axes reach down to it. That makes the decay-time factor matter: the
spurious regions sit late in the pulse, where the source is extrapolated.

Below those, each chopper has a delay slider and a *parked* toggle. The delay follows
chopcal whenever one of chopcal's inputs (lowest energy, shaping time, PSC limit)
changes; moved by hand, it stays put until one of those inputs changes. A parked disc is
stopped open, so it passes everything, as if it were not there: it is dotted in the
time-distance diagram, and its delay is greyed out.

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
pip install "chopcal>=0.8" numpy matplotlib ipympl ipywidgets jupyterlab
jupyter lab bifrost_phasespace.ipynb
```

## Files

- `bifrost_phasespace.ipynb`, `trex_phasespace.ipynb`: the notebooks.
- `ess_butterfly.py`: McStas's `ESS_butterfly` moderator for any beamline (3 cm
  moderator), as an absolute rate into (wavelength, emission time) bins. It is
  independent of BIFROST and of the rest, and `python ess_butterfly.py` checks it against
  McStas's own intensities into BIFROST's guide.
- `phasespace.py`: the calculation and the plotting for any instrument, in layers usable
  on their own:
  - `Instrument`, what an instrument must say: its beamline, guide entrance, sample
    distance, discs and how its settings set them, and its controls;
  - `transmission(instrument, settings, ...)` computes one setting and returns plain
    arrays, no plotting;
  - `make_axes()` and `draw(result, axes)` draw a result with matplotlib, for a script
    or a saved picture;
  - `draw_time_distance(result, ax)` draws only the time-distance diagram;
  - `figure(instrument, ...)` wires both to ipywidgets controls.
- `bifrost.py`: BIFROST (beamline W4), its discs set by `chopcal.bifrost`.
- `bifrost_guide_transmission.csv`: BIFROST's guide transmission against wavelength.
- `trex.py`: T-REX (beamline W7), its discs phased as `trex_primary.instr` does.

`python phasespace.py`, `python bifrost.py` and `python trex.py` run the self-checks.

## Without Jupyter

```python
import matplotlib.pyplot as plt
from bifrost import BIFROST
from phasespace import draw, make_axes, transmission

result = transmission(BIFROST, dict(energy_min=4.5, shaping_time=2e-4))   # meV, s
fig, axes = make_axes()
draw(result, axes)
fig.savefig('bifrost.png')
```
