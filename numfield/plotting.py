"""
Plotting utilities for field visualization.

This module provides helper functions for creating appropriate colormap normalizations
and color schemes for visualizing scalar fields on Cartesian meshes.

Key Features
------------
- Adaptive normalization (linear, two-slope, boundary)
- Support for discrete and continuous colormaps
- Diverging colormap presets for comparative visualization

Examples
--------
>>> from field.plotting import _get_norm, get_comparaison_cmap
>>> norm = _get_norm(vmin=0, vmax=100, vcenter=None, cbar_nb_levels=None, cmap='viridis')
>>> cmap = get_comparaison_cmap(N=256)  # Blue-White-Red diverging colormap

Note
----
This module requires matplotlib. Install it with: pip install numfield[all]
"""

import numpy as np


def _check_matplotlib():
    """Check if matplotlib is available and raise ImportError if not."""
    try:
        from matplotlib.colors import BoundaryNorm, TwoSlopeNorm, Normalize
        import matplotlib.colors as mcolors
        return BoundaryNorm, TwoSlopeNorm, Normalize, mcolors
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting utilities. "
            "Install it with: pip install numfield[all]"
        )

def _get_norm(vmin, vmax, vcenter, cbar_nb_levels, cmap):
    """
    Choose and construct the appropriate colormap normalization
    for a given data range and colorbar configuration.

    Parameters
    ----------
    vmin : float
        Minimum data value to map to the colormap.
    vmax : float
        Maximum data value to map to the colormap.
    vcenter : float or None
        Central reference value. If provided:
        - For continuous data, use `TwoSlopeNorm` for diverging colormaps.
        - For discrete data, split bins around the center.
    cbar_nb_levels : int or None
        Number of discrete color levels for the colorbar.
        If None, a continuous normalization is used.
    cmap : matplotlib.colors.Colormap
        The colormap instance for which the normalization is built.

    Returns
    -------
    norm : matplotlib.colors.Normalize
        A normalization object suitable for use with Matplotlib
        plotting functions such as `pcolormesh` or `imshow`.

    Notes
    -----
    - If `cbar_nb_levels` is provided:
        - A `BoundaryNorm` is constructed, splitting the [vmin, vmax] range
          into discrete bins.
        - If `vcenter` is also given, bins are symmetrically defined
          around the center.
    - If `cbar_nb_levels` is None:
        - A `Normalize` is used for simple linear scaling,
          unless `vcenter` is specified, in which case a `TwoSlopeNorm`
          is used for diverging data.
    """
    BoundaryNorm, TwoSlopeNorm, Normalize, _ = _check_matplotlib()

    if cbar_nb_levels is not None:
        # discrete bins
        if vcenter is not None:
            # split bins around the center
            n_lower = cbar_nb_levels // 2
            n_upper = cbar_nb_levels - n_lower
            lower_edges = np.linspace(vmin, vcenter, n_lower + 1)
            upper_edges = np.linspace(vcenter, vmax, n_upper + 1)[1:]  # skip duplicate center
            levels = np.concatenate([lower_edges, upper_edges])
            norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        else:
            # regular discrete
            levels = np.linspace(vmin, vmax, cbar_nb_levels + 1)
            norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    else:
        # continuous
        if vcenter is not None:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
    return norm

def get_comparaison_cmap(N=100):
    _, _, _, mcolors = _check_matplotlib()
    colors = ['blue', 'white', 'red']
    n_bins = 100
    return mcolors.LinearSegmentedColormap.from_list('RdBu_white', colors, N=n_bins)