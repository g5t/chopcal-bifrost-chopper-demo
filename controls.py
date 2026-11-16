"""Controls described without a GUI toolkit: what an instrument or source offers, which a
front end -- ipywidgets in a notebook (`ipywidget`), Gtk in `gtk_app` -- turns into its
own widgets. Four kinds cover every setting the demo has: a number, a range of numbers,
a choice among values, and an on/off switch.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Number:
    """A number from `min` to `max` in steps of `step`; with `log`, the steps are in
    log10 of the value, so `step=0.05` is about 12 % each."""
    label: str
    value: float
    min: float
    max: float
    step: float
    log: bool = False
    format: str = '.2f'


@dataclass
class Range:
    """A pair of numbers, (low, high), each from `min` to `max` in steps of `step`."""
    label: str
    value: tuple
    min: float
    max: float
    step: float
    format: str = '.2f'


@dataclass
class Choice:
    """One of `options`: values, or (label, value) pairs. `buttons` asks for them side by
    side, where there are few, rather than along a slider."""
    label: str
    options: list
    value: object
    buttons: bool = False

    def pairs(self) -> list:
        """The options as (label, value) pairs."""
        return [o if isinstance(o, tuple) else (str(o), o) for o in self.options]


@dataclass
class Switch:
    """On or off."""
    label: str
    value: bool


def ipywidget(control, **common):
    """The ipywidgets widget for a control; `common` sets attributes every widget shares,
    e.g. `style` and `layout`."""
    import numpy as np
    from ipywidgets import (Checkbox, FloatLogSlider, FloatRangeSlider, FloatSlider,
                            SelectionSlider, ToggleButtons)
    if isinstance(control, Number) and control.log:
        widget = FloatLogSlider(value=control.value, min=np.log10(control.min),
                                max=np.log10(control.max), step=control.step,
                                readout_format=control.format, description=control.label)
    elif isinstance(control, Number):
        widget = FloatSlider(value=control.value, min=control.min, max=control.max,
                             step=control.step, readout_format=control.format,
                             description=control.label)
    elif isinstance(control, Range):
        widget = FloatRangeSlider(value=control.value, min=control.min, max=control.max,
                                  step=control.step, readout_format=control.format,
                                  description=control.label)
    elif isinstance(control, Choice):
        kind = ToggleButtons if control.buttons else SelectionSlider
        widget = kind(options=control.options, value=control.value,
                      description=control.label)
    elif isinstance(control, Switch):
        widget = Checkbox(value=control.value, description=control.label)
    else:
        raise TypeError(f'not a control: {control!r}')
    for name, value in common.items():
        setattr(widget, name, value)
    return widget
