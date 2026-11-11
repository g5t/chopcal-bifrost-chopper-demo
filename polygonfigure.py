import uuid
from functools import partial
from typing import Literal
import scipp as sc
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from plopp import backends
from plopp.graphics import GraphicalView
from plopp.graphics.bbox import BoundingBox, axis_bounds

import matplotlib

_IPYMPL = {'widget', 'ipympl', 'module://ipympl.backend_nbagg'}
  
def ipympl_active() -> bool:
    """True if ipympl is the current backend. No side effects, no resolution."""
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
    if ip is None:              # plain python, not IPython
        return False
    if ipympl_active():
        return True
    ip.run_line_magic('matplotlib', 'widget')   # or: ip.enable_matplotlib('widget')
    if not quiet:
        print('[polygonfigure] enabled %matplotlib widget')
    return True


class Patch:
    """One matplotlib patch per element of a 1-D DataArray"""
    
    def __init__(self, *, canvas, data, uid=None, artist_number=0,
                 colormapper=None, mask_color=None, cbar=False,
                 x='x', y='y', n='n', **kwargs):
        self.uid = uid or uuid.uuid4().hex
        self._ax = canvas.ax
        self._data, self._x, self._y, self._n = data, x, y, n
        self._colormapper = colormapper
        self.label = data.name if not cbar else None
        self._coll = PatchCollection(
            self._make_patches(data), label=self.label,
            **({} if cbar else {'facecolor': f'C{artist_number}'}), **kwargs)
        self._ax.add_collection(self._coll)
        if colormapper is not None:
            colormapper.add_artist(self.uid, self)
            
    def _make_patches(self, data):
        x = data.coords[self._x].values
        y = data.coords[self._y].values
        n = data.coords[self._n].values
        return [
            Polygon(
                [(a, b) for a, b in zip(x[i==n], y[i==n])]) for i in range(1+max(n))
        ]
 
    def update(self, new_values: sc.DataArray) -> None:
        self._data = new_values
        self._coll.set_paths(self._make_patches(new_values))
        if self._colormapper is not None:
            self._update_colors()

    def bbox(self, xscale: Literal['linear', 'log'], yscale, **ignored):
        return BoundingBox(**axis_bounds(('xmin', 'xmax'), self._data.coords[self._x], xscale),
                           **axis_bounds(('ymin', 'ymax'), self._data.coords[self._y], yscale))

    def remove(self) -> None:
        self._coll.remove()
        if self._colormapper is not None:
            self._colormapper.remove_artist(self.uid)

    @property                      # these two only needed when cbar=True
    def data(self): return self._data
    def notify_artist(self, message): self._update_colors()
    def _update_colors(self): self._coll.set_facecolors(self._colormapper.rgba(self.data))


def patchfigure(*nodes, x='x', y='y', cbar=False, **kwargs):
    view_maker = partial(GraphicalView, dims={'x': x, 'y': y},
                         canvas_maker=backends.get(group='2d', name='canvas'),
                         artist_maker=Patch, colormapper=cbar)
    return backends.get(group='2d', name='figure')(view_maker, *nodes, x=x, y=y, cbar=cbar, **kwargs)


class WavelengthEnergyPatch(Patch):
    def __init__(self, **kwargs):
        from matplotlib.pyplot import setp
        super().__init__(**kwargs)
        setp(
            self._ax.secondary_yaxis('right', (bare_wavelength_to_energy, bare_energy_to_wavelength)), 
            ylabel=r'E [meV]'
        )


def wavelength_energy_patchfigure(*nodes, x='x', y='y', cbar=False, **kwargs):
    view_maker = partial(GraphicalView, dims={'x': x, 'y': y},
                         canvas_maker=backends.get(group='2d', name='canvas'),
                         artist_maker=WavelengthEnergyPatch, colormapper=cbar)
    return backends.get(group='2d', name='figure')(view_maker, *nodes, x=x, y=y, cbar=cbar, **kwargs)


def border_to_patch(xy: list[list[tuple[float, float]|list[float]]], x: dict | None = None, y: dict | None = None, dim: str | None = None):
    from scipp import array, concat, ones, DataArray
    if dim is None:
        dim = 'points'
    if x is None:
        x = {}
    if y is None:
        y = {}
        
    xs = concat([array(values=[x for x, _ in border], dims=[dim], unit=x.get('unit')) for border in xy], dim=dim)
    ys = concat([array(values=[y for _, y in border], dims=[dim], unit=y.get('unit')) for border in xy], dim=dim)
    ns = concat([ones(sizes={dim: len(border)}, dtype='int') * index for index, border in enumerate(xy)], dim=dim)
    
    return DataArray(ones(sizes={dim: sum(len(border) for border in xy)}), coords={x.get('name', 'x'): xs, y.get('name', 'y'): ys, 'n': ns})
    


def phase_to_patch(obj, translator=None, **kwargs):
    if translator is None:
        translator = lambda x: x
    return border_to_patch([translator(polygon.vertices) for polygon in obj.polygons], **kwargs)


def invv_time_to_time_wavelength(points):
    from scipp import array
    from scipp.constants import neutron_mass, Planck
    invv = array(values=[i for i, _ in points], unit='s/m', dims=['point'])
    wavelength = (Planck / neutron_mass * invv).to(unit='angstrom').values
    t = [i for _, i in points]
    return [(t, w) for t, w in zip(t, wavelength)]


def invv_time_to_time_wavelength_at_sample(points):
    # we want to 'forward' (1/v, t) from the source position to the sample position
    #   v = d / t  ===>  t = d / v
    from scipp import array, scalar
    from scipp.constants import neutron_mass, Planck
    SOURCE_TO_SAMPLE = scalar(162.0, unit='m')
    invv = array(values=[i for i, _ in points], unit='s/m', dims=['point'])
    wavelength = (Planck / neutron_mass * invv).to(unit='angstrom')
    t = array(values=[i for _, i in points], unit='s', dims=['point'])
    t_sample = t + SOURCE_TO_SAMPLE * invv
    return [(t, w) for t, w in zip(t_sample.values, wavelength.values)]    


def axes(array):
    return [f'axis_{i}' for i in range(array.ndim)]

    
def bare_wavelength_to_energy(wavelength):
    from scipp.constants import neutron_mass, Planck
    from scipp import array
    w = array(values=wavelength, unit='angstrom', dims=axes(wavelength))
    en = (Planck * Planck / 2 / neutron_mass / w / w).to(unit='meV')
    return en.values

def bare_energy_to_wavelength(energy):
    from scipp.constants import neutron_mass, Planck
    from scipp import array, sqrt
    en = array(values=energy, unit='meV', dims=axes(energy))
    wavelength = (Planck / sqrt(en * 2 * neutron_mass)).to(unit='angstrom')
    return wavelength.values


def region_obj(wavelength_limits_angstrom, latest_emission_msec):
    from chopcal.lib import Region
    return Region.from_wavelengths(min(wavelength_limits_angstrom), max(wavelength_limits_angstrom), 0., latest_emission_msec/1000.)

    
def bifrost_train_list(minimum_energy_meV, pulse_shaping_time_msec):
    from chopcal import bifrost
    values = bifrost(energy_min=minimum_energy_meV, shaping_time=pulse_shaping_time_msec / 1000.)
    return [values[x] for x in ('ps1', 'ps2', 'fo1', 'fo2', 'bw1', 'bw2')]

    
def bifrost_transmitted(source, train_list):
    return source.transmit(train_list)

    
def source_plot(source_polygons):
    return phase_to_patch(source_polygons, translator=invv_time_to_time_wavelength, x={'name': 'time', 'unit': 'sec'}, y={'name': 'wavelength', 'unit': 'angstrom'})


def source_forwarded_plot(source_polygons):
    return phase_to_patch(source_polygons, translator=invv_time_to_time_wavelength_at_sample, x={'name': 'time', 'unit': 'sec'}, y={'name': 'wavelength', 'unit': 'angstrom'})

    
def transmitted_plot(transmitted_polygons):
    return phase_to_patch(transmitted_polygons, translator=invv_time_to_time_wavelength, x={'name': 'time', 'unit': 'sec'}, y={'name': 'wavelength', 'unit': 'angstrom'})


def bifrost_forwarded_plot(transmitted_polygons):
    return phase_to_patch(transmitted_polygons, translator=invv_time_to_time_wavelength_at_sample, x={'name': 'time', 'unit': 'sec'}, y={'name': 'wavelength', 'unit': 'angstrom'})
    


def _bifrost_transmitted_phasespace_figure(ei_min=2.0, t_psc=0.1):
    from ipywidgets import FloatSlider, FloatRangeSlider, HBox, VBox
    from plopp import widget_node, Node
    
    t_last = FloatSlider(value=3.0, min=0, max=int(1/14*1000), description=r'$t_\text{last}$ [msec]')
    w_range = FloatRangeSlider(value=(2.0, 8.0), min=0.1, max=50., description=r'$\Delta\lambda$ [Å]')
    ei_min = FloatSlider(value=ei_min, min=0.1, max=50., description=r'$E$ [meV]')
    w_max = FloatSlider(value=5, min=0.1, max=10., description=r'$\lambda_\text{max}$ [Å]')
    t_psc = FloatSlider(value=t_psc, min=0.1, max=int(1/14*1000*175/360), description=r'$t_\text{psc}$ [msec]')

    t_last_node = widget_node(t_last)
    w_range_node = widget_node(w_range)
    region_node = Node(region_obj, wavelength_limits_angstrom=w_range_node, latest_emission_msec=t_last_node)
    
    ei_min_node = widget_node(ei_min)
    t_psi_node = widget_node(t_psc)
    bifrost_train_node = Node(bifrost_train_list, minimum_energy_meV=ei_min_node, pulse_shaping_time_msec=t_psi_node)
    
    bifrost_transmitted_node = Node(bifrost_transmitted, source=region_node, train_list=bifrost_train_node)
    fig_source_node = Node(source_plot, source_polygons=region_node)
    fig_transmitted_node = Node(transmitted_plot, transmitted_polygons=bifrost_transmitted_node)
    
    fig = wavelength_energy_patchfigure(fig_source_node, fig_transmitted_node, x='time', y='wavelength')

    fig_forward_source_node = Node(source_forwarded_plot, source_polygons=region_node)
    fig_forward_transmitted_node = Node(bifrost_forwarded_plot, transmitted_polygons=bifrost_transmitted_node)

    fig_forward = wavelength_energy_patchfigure(fig_forward_source_node, fig_forward_transmitted_node, x='time', y='wavelength')

    return HBox([fig, fig_forward, VBox([t_last, w_range, ei_min, t_psc])])


def bifrost_transmitted_phasespace_figure(ei_min=2.0, t_psc=0.1):
    if ensure_widget_backend():
        return _bifrost_transmitted_phasespace_figure(ei_min=ei_min, t_psc=t_psc)
    else:
        raise RuntimeError('Interactive mode needed -- please install IPython and ipympl')

