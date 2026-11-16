"""Every instrument's interactive figure in a Gtk 4 window of its own:

    python gtk_app.py [BIFROST | T-REX | IN5]

The same figure and controls as `app.ipynb`, from the same `phasespace.Session`, without a
browser or a Jupyter server. Needs PyGObject with Gtk 4 (conda-forge: pygobject, gtk4,
pycairo); see the README.

Each instrument is built the first time it is chosen, and then kept, so switching back
finds it as it was left. Moving a control recomputes once it has been still for
`SETTLE_MS`, so dragging a slider does not queue a calculation per step.
"""
from __future__ import annotations

import math
import sys

import gi

gi.require_version('Gtk', '4.0')
from gi.repository import GLib, Gtk  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use('Agg')   # make_axes uses pyplot; each figure gets a Gtk canvas below
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_gtk4 import NavigationToolbar2GTK4  # noqa: E402
from matplotlib.backends.backend_gtk4agg import FigureCanvasGTK4Agg  # noqa: E402

from bifrost import BIFROST  # noqa: E402
from controls import Choice, Number, Range, Switch  # noqa: E402
from in5 import IN5  # noqa: E402
from phasespace import Session, make_axes  # noqa: E402
from trex import TREX  # noqa: E402

#: what the drop-down offers, in order, each with its starting settings (as app.ipynb)
INSTRUMENTS = {
    'BIFROST, ESS': (BIFROST, dict(energy_min=4.5, shaping_time=0.2)),
    'T-REX, ESS': (TREX, dict(wavelength=2.4, rrm=4, mode='HF')),
    'IN5, ILL': (IN5, dict(wavelength=4.5, speed=8500, ratio=0.5, counter=True)),
}
SETTLE_MS = 150


def _digits(step):
    return max(0, -math.floor(math.log10(step))) if step < 1 else 0


def _snap(value, control):
    """As ipywidgets does: the nearest step from `min`."""
    return control.min + round((value - control.min) / control.step) * control.step


def _number(control: Number, changed):
    """A slider, and for a linear one a box to type the value in; `changed(value)`."""
    box = Gtk.Box(spacing=6)
    if control.log:   # the slider moves the exponent, and shows the value
        lo, hi = math.log10(control.min), math.log10(control.max)
        adjustment = Gtk.Adjustment(value=math.log10(control.value), lower=lo, upper=hi,
                                    step_increment=control.step,
                                    page_increment=10 * control.step)
        scale = Gtk.Scale(adjustment=adjustment, hexpand=True, draw_value=True,
                          value_pos=Gtk.PositionType.RIGHT)
        scale.set_format_value_func(lambda _, v: format(10 ** v, control.format))
        adjustment.connect('value-changed', lambda a: changed(10 ** (
            lo + round((a.get_value() - lo) / control.step) * control.step)))
        box.append(scale)
        return box, adjustment
    adjustment = Gtk.Adjustment(value=control.value, lower=control.min, upper=control.max,
                                step_increment=control.step,
                                page_increment=10 * control.step)
    box.append(Gtk.Scale(adjustment=adjustment, hexpand=True, draw_value=False))
    box.append(Gtk.SpinButton(adjustment=adjustment, digits=_digits(control.step),
                              width_chars=8))
    adjustment.connect('value-changed', lambda a: changed(_snap(a.get_value(), control)))
    return box, adjustment


def _range(control: Range, changed):
    box = Gtk.Box(spacing=6)
    ends = []
    for value in control.value:
        adjustment = Gtk.Adjustment(value=value, lower=control.min, upper=control.max,
                                    step_increment=control.step,
                                    page_increment=10 * control.step)
        box.append(Gtk.SpinButton(adjustment=adjustment, digits=_digits(control.step),
                                  hexpand=True))
        ends.append(adjustment)
    for adjustment in ends:
        adjustment.connect('value-changed', lambda _: changed(
            tuple(sorted(_snap(a.get_value(), control) for a in ends))))
    return box


def _choice(control: Choice, changed):
    pairs = control.pairs()
    values = [v for _, v in pairs]
    if control.buttons:
        box = Gtk.Box(css_classes=['linked'])
        first = None
        for label, value in pairs:
            button = Gtk.ToggleButton(label=label, active=value == control.value)
            if first is None:
                first = button
            else:
                button.set_group(first)
            button.connect('toggled', lambda b, v=value: b.get_active() and changed(v))
            box.append(button)
        return box
    menu = Gtk.DropDown.new_from_strings([label for label, _ in pairs])
    menu.set_selected(values.index(control.value))
    menu.connect('notify::selected', lambda m, _: changed(values[m.get_selected()]))
    return menu


def _switch(control: Switch, changed):
    check = Gtk.CheckButton(active=control.value)
    check.connect('toggled', lambda c: changed(c.get_active()))
    return check


def _note():
    return Gtk.Label(xalign=0, wrap=True, css_classes=['dim-label'])


class Page(Gtk.Paned):
    """One instrument: its figure, with matplotlib's toolbar, beside its controls."""

    def __init__(self, instrument, **initial):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, shrink_end_child=False,
                         resize_end_child=False, position=1150)
        self.session = session = Session(instrument, **initial)
        self.pending = None
        self.syncing = False

        fig, self.axes = make_axes()
        plt.close(fig)   # pyplot's Agg manager is not needed; the Gtk canvas below is
        self.canvas = FigureCanvasGTK4Agg(fig)
        self.canvas.set_hexpand(True)
        self.canvas.set_vexpand(True)
        left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        left.append(NavigationToolbar2GTK4(self.canvas))
        left.append(self.canvas)
        self.set_start_child(left)

        grid = Gtk.Grid(column_spacing=12, row_spacing=6, margin_start=12, margin_end=12,
                        margin_top=12, margin_bottom=12)
        row = 0

        def add(label, widget):
            nonlocal row
            grid.attach(Gtk.Label(label=label, xalign=1), 0, row, 1, 1)
            grid.attach(widget, 1, row, 2, 1)
            row += 1

        def add_wide(widget):
            nonlocal row
            grid.attach(widget, 0, row, 3, 1)
            row += 1

        self.note, self.power, self.status = _note(), _note(), _note()
        last_source = list(session.source)[-1] if session.source else None
        for name, control in session.controls.items():
            changed = self._changed(name)
            if isinstance(control, Number):
                widget, _ = _number(control, changed)
            elif isinstance(control, Range):
                widget = _range(control, changed)
            elif isinstance(control, Choice):
                widget = _choice(control, changed)
            else:
                widget = _switch(control, changed)
            add(control.label, widget)
            if name == list(session.own)[-1]:
                add_wide(self.note)
            if name == last_source:
                add_wide(self.power)
        add_wide(self.status)
        add_wide(Gtk.Label(label='<b>choppers</b>', use_markup=True, xalign=0,
                           margin_top=6))

        self.delays, self.suggested, self.delay_boxes = {}, {}, {}
        for n, control in session.delays.items():
            box, adjustment = _number(control, self._delay_changed(n))
            parked = Gtk.ToggleButton(label='parked',
                                      tooltip_text='stopped open: passes everything')
            parked.connect('toggled', self._park_changed(n))
            self.suggested[n] = Gtk.Label(xalign=0, css_classes=['dim-label'])
            grid.attach(Gtk.Label(label=control.label, xalign=1), 0, row, 1, 1)
            grid.attach(box, 1, row, 1, 1)
            grid.attach(parked, 2, row, 1, 1)
            grid.attach(self.suggested[n], 1, row + 1, 2, 1)
            row += 2
            self.delays[n], self.delay_boxes[n] = adjustment, box

        scroll = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER,
                                    min_content_width=420)
        scroll.set_child(grid)
        self.set_end_child(scroll)
        self._show_delays()
        self._redraw()

    # -- what the widgets tell the session ---------------------------------------------

    def _changed(self, name):
        def on_change(value):
            self.session.set(name, value)
            if name in self.session.own:
                self._show_delays()
            self._schedule()
        return on_change

    def _delay_changed(self, name):
        def on_change(value):
            if not self.syncing:
                self.session.set_delay(name, value)
                self._schedule()
        return on_change

    def _park_changed(self, name):
        def on_toggle(button):
            self.session.park(name, button.get_active())
            self.delay_boxes[name].set_sensitive(not button.get_active())
            self._schedule()
        return on_toggle

    # -- and what the session tells them -----------------------------------------------

    def _show_delays(self):
        self.syncing = True
        try:
            for n, control in self.session.delays.items():
                adjustment = self.delays[n]
                adjustment.set_lower(control.min)
                adjustment.set_upper(control.max)
                adjustment.set_value(control.value)
                self.suggested[n].set_label(self.session.suggested[n])
        finally:
            self.syncing = False

    def _schedule(self):
        if self.pending is not None:
            GLib.source_remove(self.pending)
        self.pending = GLib.timeout_add(SETTLE_MS, self._redraw)

    def _redraw(self):
        self.pending = None
        self.session.draw(self.axes)
        self.canvas.draw_idle()
        self.status.set_label(self.session.status)
        self.note.set_label(self.session.note)
        self.power.set_label(self.session.power)
        return GLib.SOURCE_REMOVE


class Window(Gtk.ApplicationWindow):

    def __init__(self, app, first=None):
        super().__init__(application=app, title='chopcal: choppers, interactively',
                         default_width=1700, default_height=1000)
        names = list(INSTRUMENTS)
        self.stack = Gtk.Stack()
        self.set_child(self.stack)
        choose = Gtk.DropDown.new_from_strings(names)
        header = Gtk.HeaderBar()
        header.pack_start(choose)
        self.set_titlebar(header)
        choose.connect('notify::selected', lambda d, _: self.show(names[d.get_selected()]))
        key = lambda text: text.replace('-', '').upper()
        start = next((n for n in names if first and key(n).startswith(key(first))), names[0])
        choose.set_selected(names.index(start))
        self.show(start)

    def show(self, name):
        if self.stack.get_child_by_name(name) is None:
            instrument, initial = INSTRUMENTS[name]
            self.stack.add_named(Page(instrument, **initial), name)
        self.stack.set_visible_child_name(name)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    app = Gtk.Application(application_id='eu.ess.chopcal.demo')
    app.connect('activate', lambda a: Window(a, argv[1] if len(argv) > 1 else None).present())
    return app.run(argv[:1])


if __name__ == '__main__':
    sys.exit(main())
