"""
Example meshes and fields for quick testing and demonstrations.

This module provides factory functions that generate common Cartesian meshes
and associated fields with simple, well-defined patterns. All functions return
new instances on each call.

Available Functions
-------------------
- ``mesh_1d``, ``mesh_2d``, ``mesh_3d``: Uniform Cartesian meshes
- ``field_constant``: Uniform constant field
- ``field_gaussian``: Multi-dimensional Gaussian distribution
- ``field_sinusoidal``: Product of sinusoidal waves
- ``field_random``: Random field (uniform or normal distribution)
- ``fields_collection``: Container with multiple example fields

Examples
--------
>>> from field.examples import mesh_2d, field_gaussian
>>> mesh = mesh_2d(nx=20, ny=30)
>>> field = field_gaussian(mesh, amplitude=5.0)
"""

import numpy as np
from .mesh import CartesianMesh
from .fields import CartesianField, Fields

# ─────────────────────────────────────────────────────────────────
# Mesh factories
# ─────────────────────────────────────────────────────────────────

def mesh_1d(n_cells: int = 10, length: float = 1.0, name: str = "x") -> CartesianMesh:
    """
    Create a simple 1D uniform Cartesian mesh.

    Parameters
    ----------
    n_cells : int, default=10
        Number of cells along the axis.
    length : float, default=1.0
        Total physical length of the mesh.
    name : str, default="x"
        Optional name for the axis (also used to create a dynamic attribute).

    Returns
    -------
    CartesianMesh
        A uniform 1D mesh spanning [0, length] with `n_cells` cells.
    """
    return CartesianMesh.from_linspace(
        starts=[0.0],
        stops=[length],
        n_boundaries=[n_cells + 1],
        axes_names=[name]
    )


def mesh_2d(
    nx: int = 20, ny: int = 30,
    lx: float = 1.0, ly: float = 1.5,
    x_name: str = "x", y_name: str = "y"
) -> CartesianMesh:
    """
    Create a simple 2D uniform Cartesian mesh.

    Parameters
    ----------
    nx, ny : int, default=20, 30
        Number of cells along x and y axes.
    lx, ly : float, default=1.0, 1.5
        Physical lengths along x and y axes.
    x_name, y_name : str, default="x", "y"
        Axis names.

    Returns
    -------
    CartesianMesh
        A uniform 2D mesh.
    """
    return CartesianMesh.from_linspace(
        starts=[0.0, 0.0],
        stops=[lx, ly],
        n_boundaries=[nx + 1, ny + 1],
        axes_names=[x_name, y_name]
    )


def mesh_3d(
    nx: int = 10, ny: int = 15, nz: int = 20,
    lx: float = 1.0, ly: float = 1.0, lz: float = 1.0,
    x_name: str = "x", y_name: str = "y", z_name: str = "z"
) -> CartesianMesh:
    """
    Create a simple 3D uniform Cartesian mesh.

    Parameters
    ----------
    nx, ny, nz : int
        Number of cells along x, y and z axes.
    lx, ly, lz : float
        Physical lengths along each axis.
    x_name, y_name, z_name : str
        Axis names.

    Returns
    -------
    CartesianMesh
        A uniform 3D mesh.
    """
    return CartesianMesh.from_linspace(
        starts=[0.0, 0.0, 0.0],
        stops=[lx, ly, lz],
        n_boundaries=[nx + 1, ny + 1, nz + 1],
        axes_names=[x_name, y_name, z_name]
    )


# ─────────────────────────────────────────────────────────────────
# Field factories (intensive by default)
# ─────────────────────────────────────────────────────────────────

def field_constant(
    mesh: CartesianMesh,
    value: float = 1.0,
    name: str = "constant",
    intensive: bool = True
) -> CartesianField:
    """
    Create a field with uniform constant values.

    Parameters
    ----------
    mesh : CartesianMesh
        Mesh on which the field is defined.
    value : float, default=1.0
        Constant value assigned to all cells.
    name : str, default="constant"
        Field name.
    intensive : bool, default=True
        Intensive/extensive flag.

    Returns
    -------
    CartesianField
        Uniform constant field.
    """
    values = np.full(mesh.shape, value, dtype=float)
    return CartesianField(name, mesh, values, intensive)


def field_gaussian(
    mesh: CartesianMesh,
    center: tuple | None = None,
    sigma: tuple | None = None,
    amplitude: float = 1.0,
    name: str = "gaussian",
    intensive: bool = True
) -> CartesianField:
    """
    Create a multi-dimensional Gaussian field centered on the mesh.

    The Gaussian is defined as:
        f(x) = amplitude * exp(-Σ ((x_i - c_i)^2) / (2 σ_i^2))

    Parameters
    ----------
    mesh : CartesianMesh
        Mesh on which the field is defined.
    center : tuple of float, optional
        Center coordinates (x0, y0, ...). Default is mesh geometric center.
    sigma : tuple of float, optional
        Standard deviations along each axis. Default is 0.2 * mesh size.
    amplitude : float, default=1.0
        Peak value at the center.
    name : str, default="gaussian"
        Field name.
    intensive : bool, default=True
        Intensive/extensive flag.

    Returns
    -------
    CartesianField
        Gaussian field.
    """
    centers = mesh.centers  # tuple of ndarrays, one per axis

    if center is None:
        center = tuple(0.5 * (ax[0] + ax[-1]) for ax in mesh.axes)
    if sigma is None:
        sigma = tuple(0.2 * (ax[-1] - ax[0]) for ax in mesh.axes)

    if len(center) != mesh.ndim or len(sigma) != mesh.ndim:
        raise ValueError("center and sigma must have one value per mesh axis")

    exponent = sum(
        ((c - c0) ** 2) / (2 * s0 ** 2)
        for c, c0, s0 in zip(centers, center, sigma)
    )
    values = amplitude * np.exp(-exponent)
    return CartesianField(name, mesh, values, intensive)


def field_sinusoidal(
    mesh: CartesianMesh,
    frequencies: tuple | None = None,
    phase: tuple | None = None,
    amplitude: float = 1.0,
    name: str = "sinusoidal",
    intensive: bool = True
) -> CartesianField:
    """
    Create a multi-dimensional sinusoidal field.

    Parameters
    ----------
    mesh : CartesianMesh
        Mesh on which the field is defined.
    frequencies : tuple of float, optional
        Wave numbers (2π / wavelength) along each axis. Default is 1.0 per axis.
    phase : tuple of float, optional
        Phase offsets along each axis. Default is 0 for all axes.
    amplitude : float, default=1.0
        Amplitude of the wave.
    name : str, default="sinusoidal"
        Field name.
    intensive : bool, default=True
        Intensive/extensive flag.

    Returns
    -------
    CartesianField
        Sinusoidal field: amplitude * ∏ sin(2π freq_i * x_i + phase_i).
    """
    centers = mesh.centers
    ndim = mesh.ndim

    if frequencies is None:
        frequencies = tuple(1.0 for _ in range(ndim))
    if phase is None:
        phase = tuple(0.0 for _ in range(ndim))

    if len(frequencies) != ndim or len(phase) != ndim:
        raise ValueError("frequencies and phase must match mesh dimensionality")

    wave = 1.0
    for c, freq, ph in zip(centers, frequencies, phase):
        wave *= np.sin(2 * np.pi * freq * c + ph)

    values = amplitude * wave
    return CartesianField(name, mesh, values, intensive)


def field_random(
    mesh: CartesianMesh,
    seed: int | None = None,
    distribution: str = "uniform",
    name: str = "random",
    intensive: bool = True
) -> CartesianField:
    """
    Create a field with random values.

    Parameters
    ----------
    mesh : CartesianMesh
        Mesh on which the field is defined.
    seed : int or None, optional
        Random seed for reproducibility.
    distribution : {"uniform", "normal"}, default="uniform"
        Type of random distribution:
        - "uniform": values in [0, 1)
        - "normal": standard normal distribution (mean=0, std=1).
    name : str, default="random"
        Field name.
    intensive : bool, default=True
        Intensive/extensive flag.

    Returns
    -------
    CartesianField
        Random field.

    Raises
    ------
    ValueError
        If `distribution` is not one of the supported options.
    """
    rng = np.random.default_rng(seed)
    shape = mesh.shape

    match distribution:
        case "uniform":
            values = rng.random(shape)
        case "normal":
            values = rng.standard_normal(shape)
        case _:
            raise ValueError(f"Unsupported distribution '{distribution}'. Use 'uniform' or 'normal'.")

    return CartesianField(name, mesh, values, intensive)


# ─────────────────────────────────────────────────────────────────
# Composite example: a Fields container with common test fields
# ─────────────────────────────────────────────────────────────────

def fields_collection(mesh: CartesianMesh | None = None) -> Fields:
    """
    Create a Fields container with a collection of common test fields.

    The generated fields cover a range of shapes and behaviours:
    constant, Gaussian, sinusoidal and random. All are intensive by default.

    Parameters
    ----------
    mesh : CartesianMesh, optional
        Mesh to use for all fields. If None, a default 2D mesh is created.

    Returns
    -------
    Fields
        Container populated with example fields.
    """
    if mesh is None:
        mesh = mesh_2d()

    container = Fields(mesh)
    container.add_values("constant", field_constant(mesh).values)
    container.add_values("gaussian", field_gaussian(mesh).values)
    container.add_values("sinusoidal", field_sinusoidal(mesh).values)
    container.add_values("random", field_random(mesh, seed=42).values)
    return container

# Public API
__all__ = [
    "mesh_1d",
    "mesh_2d",
    "mesh_3d",
    "field_constant",
    "field_gaussian",
    "field_sinusoidal",
    "field_random",
    "fields_collection",
]
