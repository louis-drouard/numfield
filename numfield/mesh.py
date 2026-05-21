"""
Cartesian mesh module for structured N-dimensional grids.

This module provides the :class:`CartesianMesh` class for creating and manipulating
structured Cartesian meshes in arbitrary dimensions, along with utility functions
for merging meshes.

Key Features
------------
- N-dimensional uniform and non-uniform Cartesian meshes
- Mesh operations: rotation, transposition, dimension reduction
- Nested mesh checking and projection support
- HDF5 I/O for persistent storage

Examples
--------
>>> from field.mesh import CartesianMesh
>>> mesh = CartesianMesh([1.0, 1.0], [2.0, 2.0])  # 2D mesh with dx=dy=1
>>> mesh.shape
(2, 2)
>>> mesh.volume
4.0
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py

from functools import reduce
import operator
from copy import deepcopy
from typing import Protocol
import logging

logger = logging.getLogger(__name__)

FLOATING_MESH_PRECISION = 8
ATOL_FLOATING_MESH_PRECISION = 10**(-FLOATING_MESH_PRECISION)

def isin_close(a, b, atol=1e-8):
    """
    Return True if every element of `a` is in `b` 
    """
    import numpy as np

    a = np.asarray(a)
    b = np.asarray(b)

    if b.size == 0:
        return a.size == 0

    b_sorted = np.sort(b)

    idx = np.searchsorted(b_sorted, a)

    idx_left = np.clip(idx - 1, 0, len(b_sorted) - 1)
    idx_right = np.clip(idx, 0, len(b_sorted) - 1)

    left = b_sorted[idx_left]
    right = b_sorted[idx_right]

    mask = (
        np.abs(a - left) <= atol
    ) | (
        np.abs(a - right) <= atol
    )
    return np.all(mask)

class CartesianMesh:
    """
    Represents a Cartesian structured mesh in N-Dimensions.

    A CartesianMesh is defined by:
    - A set of cell widths (`deltas`) along each axes.
    - An origin specifying the lower bound of the mesh.
    - Derived properties such as cell centers, volumes, and bounding box.

    This class provides utilities for mesh manipulation, such as
    dimension reduction, rotations, and checking whether another
    mesh is nested in it.

    Parameters
    ----------
    *deltas : array-like of shape (n_i,)
        Cell widths along each spatial axes.
        For example, in 2D: (dx, dy), in 3D: (dx, dy, dz).
    origin : array-like of shape (ndim,), optional
        Coordinates of the lower corner of the mesh.
        If not provided, defaults to the origin at (0, 0, ...).
    axes_names: names of the axes, default: x, y, z, u, v ,w , ...

    Raises
    ------
    ValueError
        If the length of `origin` does not match the number of dimensions.

    Examples
    --------
    >>> mesh = CartesianMesh([1.0, 1.0, 2.0], [0.5, 0.5])
    >>> mesh.shape
    (3, 2)
    >>> mesh.center
    (1.5, 0.75)
    """

    def __init__(self, *deltas, origin=None, axes_names=None, name="") -> None:
        self._ndim = len(deltas)
        self._deltas = tuple(np.round(delta, FLOATING_MESH_PRECISION) for delta in deltas)
        origin = np.zeros(self._ndim) if origin is None else origin
        if len(origin) != self._ndim: # type: ignore
            raise ValueError(f'Length of origin ({len(origin)}) is not compatible with dimension ({self._ndim})') # type: ignore
        self._axes = tuple(
            np.round(np.insert(o + np.cumsum(delta), 0, o), FLOATING_MESH_PRECISION) for delta, o in zip(deltas, origin)  # type: ignore
        )
        self.axes_names = axes_names if axes_names else self._default_axes_names()
        if self.axes_names:
            for i, ax_name in enumerate(self.axes_names):
                setattr(self, ax_name, self._axes[i])
                setattr(self, "d"+ax_name, self._deltas[i])
        self.name = name

    @classmethod
    def from_axes(cls, *axes, axes_names=None, name=""):
        """
        Construct a mesh from explicit axes coordinates.

        Parameters
        ----------
        *axes : array-like
            Sequences of coordinates defining the cell boundaries
            along each dimension.

        Returns
        -------
        CartesianMesh
            A new mesh whose cell widths and origin are inferred from `axes`.
        """

        rounded_unique_axes = tuple(np.unique(np.round(ax,8)) for ax in axes)
        return cls(*tuple(np.diff(ax) for ax in rounded_unique_axes), origin=tuple(ax[0] for ax in rounded_unique_axes), axes_names=axes_names, name=name) # type: ignore

    @classmethod
    def from_linspace(cls, starts:list[float], stops:list[float], n_boundaries: list[int], axes_names=None, name=""):
        """
        Construct a mesh from linearly-spaced boundaries along each axis.

        Parameters
        ----------
        starts : list of float
            Starting coordinate for each axis.
        stops : list of float
            Ending coordinate for each axis.
        n_boundaries : list of int
            Number of boundary points along each axis.
        axes_names : list of str, optional
            Names for each axis.

        Returns
        -------
        CartesianMesh
            A new mesh constructed from the specified linspaces.

        Raises
        ------
        ValueError
            If the lengths of `starts`, `stops`, and `n_boundaries` are inconsistent,
            or if `axes_names` length does not match number of dimensions.
        """
        if len(set((len(starts), len(stops), len(n_boundaries)))) != 1:
            raise ValueError(f'Inconsistent dimensions between the number of cells {len(n_boundaries)} and the mesh starts {len(starts)} and stops {len(stops)}')
        if axes_names is not None and len(axes_names) != len(starts):
            raise ValueError(f'Inconsistent number of axes {len(starts)} and axes names {len(axes_names)}')
        axes = [np.linspace(start, stop, n) for start, stop, n in zip(starts, stops, n_boundaries)]
        return cls.from_axes(*axes, axes_names=axes_names, name=name)

    @classmethod
    def from_arange(cls, starts:list[float], stops:list[float], steps:list[float], axes_names=None, name=""):
        """
        Construct a mesh from arange-style boundaries along each axis.

        Parameters
        ----------
        starts : list of float
            Starting coordinate for each axis.
        stops : list of float
            Stopping coordinate for each axis.
        steps :list of float
            Step sizes along each axis.
        axes_names : list of str, optional
            Names for each axis.

        Returns
        -------
        CartesianMesh
            A new mesh constructed from the specified aranges.

        Raises
        ------
        ValueError
            If argument lengths are inconsistent or ``axes_names`` length does not
            match the number of dimensions.

        Examples
        --------

        >>> mesh = CartesianMesh.from_arange([0.0, 0.0], [10.0, 20.0], [0.5, 1.0])

        """

        if len({(len(starts), len(stops), len(steps))}) != 1:
            raise ValueError(f'Inconsistent dimensions between the number of steps {len(steps)} and the mesh starts {len(starts)} and stops {len(stops)}')
        if axes_names is not None and len(axes_names) != len(starts):
            raise ValueError(f'Inconsistent number of axes {len(starts)} and axes names {len(axes_names)}')
        axes = [np.arange(start, stop, s) for start, stop, s in zip(starts, stops, steps)]
       
        return cls.from_axes(*axes, axes_names=axes_names, name=name)
    
    def _default_axes_names(self)->None|list[str]:
        """
        Generate default axis names for the mesh based on its dimensionality.

        Returns
        -------
        list of str or None
            A list of default axis names (e.g., ['x', 'y', 'z'] for 3D),
            or None if the dimensionality exceeds the supported names.
        """
        base = ['x', 'y', 'z', 'u', 'v', 'w', 'i', 'j', 'k', 'l', 'm', 'n']
        if self.ndim > len(base):
            logger.error(f'axis name is not supported for high dimension {self.ndim}')
            return None
        return base[:self.ndim]
 
    @property
    def deltas(self):
        """tuple of ndarray: Cell widths along each axes."""
        return self._deltas
    
    @property
    def axes(self):
        """tuple of ndarray: Coordinates of mesh cell boundaries along each axes."""
        return self._axes
    
    @property
    def ndim(self):
        """int: Number of spatial dimensions."""
        return self._ndim
    
    @property
    def shape(self):
        """tuple of int: Number of cells along each axes."""
        return tuple(len(ax)-1 for ax in self._axes)

    @property
    def size(self):
        """tuple of float: Total physical size of the mesh along each axes."""
        return tuple(ax[-1]-ax[0] for ax in self._axes)

    @property
    def origin(self):
        """tuple of float: Coordinates of the lower corner of the mesh."""
        return tuple(ax[0] for ax in self._axes)

    @property
    def bounding_box(self):
        """
        tuple of (float, float): Bounding box of the mesh for each axes.

        Returns
        -------
        ((xmin, xmax), (ymin, ymax), (zmin, zmax), ...)
        """
        return tuple((float(ax[0]),float(ax[-1])) for ax in self._axes)

    @property
    def center(self):
        """
        tuple of float: Coordinates of the geometric center of the mesh.
        """
        center = tuple((ax[0] + ax[-1]) / 2.0 for ax in self._axes)
        return center

    @property
    def volume(self):
        """float: Total volume (or length/area in lower dimensions) of the mesh."""
        volume = np.prod(self.size)
        return volume
    
    @property
    def centers(self):
        """
        tuple of ndarray: Coordinates of cell centers as meshgrids.

        Returns
        -------
        meshgrid
            Meshgrid arrays for cell centers in each dimension.
        """
        ctrs = [0.5 * (ax[:-1] + ax[1:]) for ax in self._axes]
        return np.meshgrid(*ctrs, indexing="ij")
    
    @property
    def axes_min(self):
        """
        tuple of ndarray: Meshgrid arrays of the minimum coordinates of each cell.

        Returns
        -------
        tuple of ndarray
            Cell minimum coordinates as meshgrid arrays.
        """
        return np.meshgrid(*[ax[:-1] for ax in self._axes], indexing="ij")
    
    @property
    def axes_max(self):
        """
        tuple of ndarray: Meshgrid arrays of the maximum coordinates of each cell.

        Returns
        -------
        tuple of ndarray
            Cell maximum coordinates as meshgrid arrays.
        """
        return np.meshgrid(*[ax[1:] for ax in self._axes], indexing="ij")
    
    @property
    def volumes(self):
        """
        ndarray: Volume (or length/area in lower dimensions) of each cell.

        Computed as the product of cell widths along each axes.
        """
        volumes = reduce(operator.mul, np.ix_(*self._deltas))
        return volumes
   
    def is_contained_in(self, mesh):
        """
        Check if this mesh bounding box is contained in the bounding box of another mesh.

        Parameters
        ----------
        mesh : CartesianMesh
            Mesh to compare against.

        Returns
        -------
        bool
            True if bounding box of `self` is contained in the bounding box of `mesh`
        """
        return all(
            ax1_min >= ax2_min and ax1_max <= ax2_max
            for (ax1_min, ax1_max), (ax2_min, ax2_max) in zip(self.bounding_box, mesh.bounding_box)
            )
       
    def is_submesh_of(self, mesh):
        """
        Check if this mesh is topologically nested in another mesh. Meaning if its axes values are all contained in the axes values of the other mesh

        Parameters
        ----------
        mesh : CartesianMesh
            Mesh to compare against.

        Returns
        -------
        bool
            True if all cell boundaries of `mesh` lie on boundaries of `self`.
        """
        if mesh.ndim != self._ndim:
            return False
        return all(isin_close(self_axes, mesh_axes, atol=ATOL_FLOATING_MESH_PRECISION) for self_axes, mesh_axes in zip(self._axes, mesh.axes))
        # return all(
        #     np.isin(mesh_axes, self_axes).all()
        #     for self_axes, mesh_axes in zip(self._axes, mesh.axes)
        # )

    def is_overlapping(self, mesh):
        """
        Check if this mesh is overlapping with another mesh.

        Parameters
        ----------
        mesh : CartesianMesh
            Mesh to compare against.

        Returns
        -------
        bool
            True if any cell of `mesh` lie with any cell of `self`.
        """
        if mesh.ndim != self._ndim:
            return False
        
        bbox1 = self.bounding_box
        bbox2 = mesh.bounding_box
        not_overlap = any(
            (a_max <= b_min) or (b_max <= a_min)
            for (a_min, a_max), (b_min, b_max) in zip(bbox1, bbox2)
            )
        overlap = not not_overlap
        return overlap

    def is_regular(self):
        """
        Check if the mesh has uniform spacing along all axes.

        Returns
        -------
        bool
            True if all cell widths are constant within tolerance along each axis.
        """
        return all([np.allclose(d[1:], d[0], atol=ATOL_FLOATING_MESH_PRECISION) for d in self.deltas])

    def copy(self):
        """
        Create a deep copy of the mesh.

        Returns
        -------
        CartesianMesh
            A copy of the current mesh.
        """
        return deepcopy(self)

    def projected_on(self, target) -> 'CartesianMesh':
        """
        Create a new mesh combining axes from target mesh
        and not defined (in target mesh) current mesh axes.

        Example
        --------
        >>> mesh3d = CartesianMesh.from_axes(x_fine, y_fine, z_fine)
        >>> mesh2d = CartesianMesh.from_axes(x_coarse, y_coarse)
        >>> mesh_proj = mesh3d.projected_on(mesh2d)
        # -> (x_coarse, y_coarse, z_fine)

        Parameters
        -----------
        target : CartesianMesh
            Target mesh, with smaller or equal dimension.

        Returns
        ---------
        CartesianMesh
            New combined mesh.
        """

        if target.ndim > self.ndim:
            raise ValueError(
                f"Target Mesh ({target.ndim}D) cannot have more dimensions "
                f"than current mesh ({self.ndim}D)."
            )
        
        # Combine axes : target mesh first, then the remaining ones
        new_axes = list(target.axes) + list(self.axes[target.ndim:])
        return CartesianMesh.from_axes(*new_axes)

    def drop_dimension(self, reduced_axes: int | list[int], keepdims=False)->'CartesianMesh':
        """
        Create a lower-dimensional mesh by removing one or more axes.

        Parameters
        ----------
        reduced_axes : int or list of int
            Indices of the axes to remove.

        Returns
        -------
        CartesianMesh
            A new mesh with fewer dimensions.

        Raises
        ------
        ValueError
            If any specified axes index is invalid.
        """
        if isinstance(reduced_axes, int):
            reduced_axes = [reduced_axes]
        if not all(isinstance(ax, int) and ax >= 0 and ax<self.ndim for ax in reduced_axes):
            raise ValueError(f"reduced_axes must be an integer between 0 (included) and mesh dimension {self._ndim} (excluded)")
        if keepdims:
            return CartesianMesh.from_axes(
                *[axis if i not in reduced_axes else [axis[0], axis[-1]]for i, axis in enumerate(self._axes)]
                )
        new_mesh = CartesianMesh(
            *[delta for i, delta in enumerate(self._deltas) if i not in reduced_axes],
            origin=tuple(origin for i, origin in enumerate(self.origin) if i not in reduced_axes),
            axes_names = [name for i, name in enumerate(self.axes_names) if i not in reduced_axes] if self.axes_names else None
            )
        return new_mesh
    
    def transpose(self, axes=None)->'CartesianMesh':
        """
        Transpose the mesh axes.

        Parameters
        ----------
        axes : tuple of int, optional
            New ordering of the axes. If None, reverses the axes.
            Supports negative indexing (e.g., -1 for last axis).

        Returns
        -------
        CartesianMesh
            A new mesh with transposed axes.

        Raises
        ------
        ValueError
            If `axes` is not a permutation of dimension indices,
            or if its length does not match the mesh dimensionality.
        """

        ndim = self._ndim
        if axes is None:
            axes = tuple(reversed(range(ndim)))
        else:
            if len(axes) != ndim:
                raise ValueError(f"axes must have exactly {ndim} elements for {ndim}D mesh")
            # Normalize negative indices
            axes = tuple(ax % ndim for ax in axes)
            if len(set(axes)) != ndim:
                raise ValueError("axes must be a permutation of all dimension indices (0 to ndim-1)")

        # Reorder axes, preserving the stored arrays
        new_axes = tuple(self._axes[i] for i in axes)
        new_axes_names = [self.axes_names[i] for i in axes] if self.axes_names else None

        # Build new mesh from transposed axes
        return CartesianMesh.from_axes(*new_axes, axes_names=new_axes_names)

    def rot90(self, nb_rotation: int, axes: tuple[int, int]= (0,1)):
        """
        Rotate the mesh by multiples of 90° in a given plane.

        The rotation is topological: the origin remains unchanged,
        but the order of cell widths is modified.

        Parameters
        ----------
        nb_rotation : int
            Number of 90° rotations (modulo 4).
        axes : tuple of int
            Pair of axes indices defining the rotation plane.

        Returns
        -------
        CartesianMesh
            Rotated mesh.

        Raises
        ------
        ValueError
            If `axes` is invalid.
        """
        if len(axes) != 2:
            raise ValueError("axes must be a tuple of exactly two indices")

        i, j = axes
        if i == j or not (0 <= i < self._ndim) or not (0 <= j < self._ndim):
            raise ValueError(f"Invalid plane of rotation: {axes}")       

        new_deltas = [deepcopy(delta) for delta in self._deltas]

        di = self._deltas[i]
        dj = self._deltas[j]

        nb_rotation %= 4
        match nb_rotation:
            case 0:
                return self.copy()
            case 1:  # +90°
                new_deltas[i] = dj[::-1]
                new_deltas[j] = di
            case 2:  # 180°
                new_deltas[i] = di[::-1]
                new_deltas[j] = dj[::-1]
            case 3:  # -90° (270°)
                new_deltas[i] = dj
                new_deltas[j] = di[::-1]

        return self.__class__(*new_deltas, origin=self.origin)

    def to_hdf_group(self, parent_group):
        """
        Write mesh data into an existing HDF5 group.

        Parameters
        ----------
        parent_group : h5py.Group
            Parent HDF5 group in which to create the mesh datasets.
            Creates datasets: 'origin', 'axes_names', 'shape', and a group 'axes'
            containing individual axis arrays.
        """
        parent_group.create_dataset("origin", data=self.origin)
        parent_group.create_dataset("axes_names", data=self.axes_names)
        parent_group.create_dataset("shape", data=self.shape)
        if self.name:
            parent_group.create_dataset("name", data=self.name)
        axes_grp = parent_group.create_group("axes")
        for i, ax in enumerate(self.axes):
            axes_grp.create_dataset(f"axis_{i}", data=ax)

    @classmethod
    def from_hdf_group(cls, group)->'CartesianMesh':
        """
        Rebuild a CartesianMesh from an existing HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group containing mesh data (datasets 'origin', 'axes_names',
            'shape', and subgroup 'axes' with axis arrays).

        Returns
        -------
        CartesianMesh
            The reconstructed mesh.
        """
        axes = [group["axes"][f"axis_{i}"][:] for i in range(len(group["shape"]))]
        return cls.from_axes(
            *axes,
            axes_names=[an.decode('utf-8') for an in group["axes_names"]] if "axes_names" in group else None,
            name=group["name"].decode('utf-8') if "name" in group else ""
        )

    def to_hdf(self, filename):
        """
        Save the mesh to an HDF5 file.

        Parameters
        ----------
        filename : str
            Path to the HDF5 file to create.
        """
        with h5py.File(filename, "w") as f: # pyright: ignore[reportAttributeAccessIssue]
            self.to_hdf_group(f)

    @classmethod
    def from_hdf(cls, filename: str):
        """
        Load a mesh from an HDF5 file.

        Parameters
        ----------
        filename : str
            Path to the HDF5 file to load.

        Returns
        -------
        CartesianMesh
            The loaded mesh.
        """
        with h5py.File(filename, "r") as f: # pyright: ignore[reportAttributeAccessIssue]
            return cls.from_hdf_group(f)


    def __eq__(self, value: object) -> bool:
        """Check equality between two meshes."""
        if not isinstance(value, CartesianMesh):
            return False
        if self._ndim != value.ndim or self.shape != value.shape:
            return False
        return all(np.allclose(a, b, atol=ATOL_FLOATING_MESH_PRECISION) for a, b in zip(self._axes, value._axes))

    def __repr__(self) -> str:
        """Text representation of the mesh."""
        string = f"{type(self).__name__} {self.name} {self.shape}:\n"
        if self.axes_names:
            for name, ax in zip(self.axes_names, self._axes):
                string += f"{name}: {ax}\n"
        else:
            for i, ax in enumerate(self._axes):
                string += f"x{i}: {ax}\n"
        return string[:-1]


def merge_meshes(*meshes):
    """
    Merge multiple Cartesian meshes into a single combined mesh.

    This function concatenates and unifies the coordinate axes of all input meshes
    along each dimension. It ensures that all meshes have the same dimensionality 
    before merging. Duplicate coordinates are automatically removed using `np.unique`.

    Parameters
    ----------
    *meshes : CartesianMesh
        One or more Cartesian meshes to merge. All meshes must have the same
        number of dimensions.

    Returns
    -------
    CartesianMesh
        A new mesh whose axes are the union of all input mesh axes. The specific
        class returned depends on the dimensionality and type of the first mesh.

    Raises
    ------
    ValueError
        If no meshes are provided.
    ValueError
        If the meshes do not all share the same number of dimensions.

    Examples
    --------
    >>> mesh1 = CartesianMesh.from_axes(x1, y1, z1)
    >>> mesh2 = CartesianMesh.from_axes(x2, y2, z2)
    >>> merged = merge_meshes(mesh1, mesh2)
    >>> merged.x.shape, merged.y.shape, merged.z.shape
    ((len(np.unique(np.concatenate([x1, x2]))),
      len(np.unique(np.concatenate([y1, y2]))),
      len(np.unique(np.concatenate([z1, z2]))))
    """
    if len(meshes) < 1:
        raise ValueError('You cannot combine less than 1 mesh')
    
    ndim = meshes[0].ndim
    if any(mesh.ndim != ndim for mesh in meshes):
        raise ValueError(f"All the meshes must have the same dimension, ndim of the first mesh is {ndim}")
    
    new_axes = [np.unique(np.concatenate([mesh.axes[i] for mesh in meshes])) for i in range(ndim)]
    
    return CartesianMesh.from_axes(*new_axes)
