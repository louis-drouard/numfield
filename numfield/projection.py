"""
Field projection module for conservative interpolation between meshes.

This module provides functions for projecting fields defined on one Cartesian mesh
onto another mesh using conservative remapping based on geometric overlap.

Key Features
------------
- Conservative 1D transfer matrix construction
- N-dimensional field projection via tensorized operations
- Support for both intensive (averages) and extensive (integrals) quantities
- Optional weighting for complex field combinations

Examples
--------
>>> import numpy as np
>>> from field.projection import project_ND
>>> src_mesh = (np.linspace(0, 1, 11),)  # Fine mesh
>>> src_values = np.random.rand(10)
>>> target_mesh = (np.linspace(0, 1, 6),)  # Coarse mesh
>>> projected = project_ND(src_mesh, src_values, target_mesh)
"""

import numpy as np

def build_1D_transfer_matrix(src_mesh: np.ndarray, dest_mesh: np.ndarray, intensive=True) -> np.ndarray:
    """
    Build a 1D transfer matrix to project data from one mesh to another
    using the overlap of intervals.

    Each element T[i, j] represents the fraction (or integral part, depending on
    `intensive`) of the source cell i that contributes to the target cell j.

    Parameters
    ----------
    src_mesh : np.ndarray
        Monotonic array of cell boundaries of the source mesh (size N+1).
    dest_mesh : np.ndarray
        Monotonic array of cell boundaries of the target mesh (size M+1).
    intensive : bool, default=True
        If True, normalize the overlap lengths by the width of the target cells,
        suitable for intensive quantities (e.g., density, temperature).
        If False, normalize by the width of source cells 
        (for extensive quantities like mass, energy, or integrated flux).

    Returns
    -------
    T : np.ndarray of shape (N, M)
        The transfer matrix where each entry corresponds to the proportion
        of source cell i contributing to target cell j.

    Notes
    -----
    - Both meshes must be 1D and defined by strictly increasing boundaries.
    - If meshes are not nested or aligned, partial overlaps are handled
      automatically using geometric intersections.
    - For intensive fields: target = source ⋅ T
    - For extensive fields: target = source ⋅ T, where source values 
      are per-unit-length quantities (will conserve total integral)
    """
    # cell boundaries
    s_start, s_end = src_mesh[:-1], src_mesh[1:]
    t_start, t_end = dest_mesh[:-1], dest_mesh[1:]

    # Broadcasting, compare every boundaries -> matrixes
    # 1st line compare (max/min) every boundary of target to the first boundary of src
    # 2nd line compare (max/min) every boundary of target to the second boundary of src
    overlap_start = np.maximum(s_start[:, None], t_start[None, :])
    overlap_end   = np.minimum(s_end[:, None],   t_end[None, :])

    # intersection length
    # 1 line contains the overlap length of every cell of target with the first cell of src
    # 2 line contains the overlap length of every cell of target with the second cell of src
    T = np.clip(overlap_end - overlap_start, 0, None)

    # Normalization
    if intensive:
        # Normalize by target cell size (preserve averages)
        t_widths = t_end - t_start
        T = T / t_widths[None, :]
    else:
        # Normalize by source cell size (preserve integrals)
        s_widths = s_end - s_start
        T = T / s_widths[:, None]

    return T

def project_ND(src_mesh, src_values, target_mesh, weights=None, intensive=True):
    """
    Project an N-dimensional field defined on a source mesh
    onto a target mesh using tensorized 1D transfer matrices,
    optionally weighted by a physical field.

    The projection conserves either integrated quantities (extensive)
    or cell-averaged quantities (intensive), depending on the `intensive` flag.

    Parameters
    ----------
    src_mesh : tuple of np.ndarray
        Tuple of length N with the cell boundaries of the source mesh
        for each axis (e.g., (x_src, y_src, z_src, ...)).
    src_values : np.ndarray
        N-dimensional array of field values defined on the source mesh cells.
        Its shape must match the number of cells in each source dimension.
    target_mesh : tuple of np.ndarray
        Tuple of length N with the cell boundaries of the target mesh
        for each axis.
    weights : np.ndarray, optional
        Weight field (same shape as src_values). If provided,
        the projection uses weighted contributions: 
        e.g., (src_values * weights) projected, then divided by (weights projected).
    intensive : bool, default=True
        If True, project intensive quantities (averages over cells).
        If False, project extensive quantities (e.g., total mass or energy).

    Returns
    -------
    target_values : np.ndarray
        Field values defined on the target mesh after projection.

    Raises
    ------
    ValueError
        If the dimensions of the meshes and `src_values` are inconsistent.

    Notes
    -----
    - The algorithm applies 1D projections sequentially along each axis
      using `np.tensordot`.
    - Supports arbitrary structured Cartesian meshes of different resolutions.
    """
    if not isinstance(src_mesh, tuple):
        raise ValueError(f"src_mesh (type={type(src_mesh)}) must be a tuple of length N containing mesh dimensions (X0, X1, X2, ..., XN)")
    if not isinstance(target_mesh, tuple):
        raise ValueError(f"target_mesh (type={type(target_mesh)}) must be a tuple of length N containing mesh dimensions (X0, X1, X2, ..., XN)")
    if len(src_mesh) != len(target_mesh) != len(src_values.shape):
        raise ValueError(f"src_mesh (dim={len(src_mesh)}), src_values (dim={len(src_values)}) and target_mesh (dim={len(target_mesh)}) must be of the same dimension")
    

    # Handle weights
    if weights is not None:
        if weights.shape != src_values.shape:
            raise ValueError(f"weights {weights.shape} must have the same shape as src_values {src_values.shape}")
        # Normalize weights to sum to 1 per cell? Not required — just use as-is.
        # compute weighted integral and weighted total mass
        src_weighted = src_values * weights
        total_weight_target = weights         # to be projected
    else:
        src_weighted = src_values
        total_weight_target = np.ones_like(src_values)

    transfer_matrices = [build_1D_transfer_matrix(src, trgt, intensive=intensive) for src, trgt in zip(src_mesh, target_mesh)]

    # Project weighted field and total weight
    for mat in transfer_matrices:
        src_weighted = np.tensordot(src_weighted, mat, axes=(0, 0))
        total_weight_target = np.tensordot(total_weight_target, mat, axes=(0, 0))

    # Final result: weighted average
    # Avoid division by zero
    target_values = np.divide(src_weighted, total_weight_target, out=np.zeros_like(src_weighted), where=total_weight_target != 0)

    return target_values

