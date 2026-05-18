"""Tests for the field.examples module."""

import pytest
import numpy as np
from numpy.testing import assert_array_equal

from numfield import (
    mesh_1d,
    mesh_2d,
    mesh_3d,
    field_constant,
    field_gaussian,
    field_sinusoidal,
    field_random,
    fields_collection,
)
from numfield.mesh import CartesianMesh
from numfield.fields import CartesianField, Fields


# ─────────────────────────────────────────────────────────────────
# Mesh factories
# ─────────────────────────────────────────────────────────────────

def test_mesh_1d_basic():
    m = mesh_1d()
    assert isinstance(m, CartesianMesh)
    assert m.ndim == 1
    assert m.shape == (10,)
    assert m.axes_names == ["x"]
    assert np.isclose(m.axes[0][0], 0.0)
    assert np.isclose(m.axes[0][-1], 1.0)


def test_mesh_1d_custom():
    m = mesh_1d(n_cells=25, length=2.5, name="t")
    assert m.shape == (25,)
    assert len(m.axes[0]) == 26
    assert np.isclose(m.axes[0][-1], 2.5)
    assert m.axes_names == ["t"]


def test_mesh_2d_basic():
    m = mesh_2d()
    assert isinstance(m, CartesianMesh)
    assert m.ndim == 2
    assert m.shape == (20, 30)
    assert m.axes_names == ["x", "y"]


def test_mesh_2d_custom():
    m = mesh_2d(nx=10, ny=15, lx=2.0, ly=3.0, x_name="u", y_name="v")
    assert m.shape == (10, 15)
    assert np.isclose(m.size[0], 2.0) and np.isclose(m.size[1], 3.0)
    assert m.axes_names == ["u", "v"]


def test_mesh_3d_basic():
    m = mesh_3d()
    assert isinstance(m, CartesianMesh)
    assert m.ndim == 3
    assert m.shape == (10, 15, 20)
    assert m.axes_names == ["x", "y", "z"]


def test_mesh_3d_custom():
    m = mesh_3d(nx=2, ny=3, nz=4, lx=1.0, ly=2.0, lz=3.0)
    assert m.shape == (2, 3, 4)
    assert np.isclose(m.volume, 1.0 * 2.0 * 3.0)


# ─────────────────────────────────────────────────────────────────
# Field factories
# ─────────────────────────────────────────────────────────────────

def test_field_constant():
    m = mesh_2d(nx=5, ny=5)
    f = field_constant(m, value=3.14, name="pi")
    assert isinstance(f, CartesianField)
    assert f.name == "pi"
    assert f.shape == (5, 5)
    assert f.intensive is True
    assert np.all(f.values == 3.14)


def test_field_gaussian_defaults():
    m = mesh_2d(nx=20, ny=30)
    f = field_gaussian(m)
    assert f.shape == (20, 30)
    assert f.intensive is True
    # Gaussian should peak near center
    center_idx = (10, 15)
    assert f.values[center_idx] > f.values[0, 0]


def test_field_gaussian_custom_center_sigma():
    m = mesh_2d(nx=10, ny=10, lx=1.0, ly=1.0)
    f = field_gaussian(m, center=(0.25, 0.75), sigma=(0.1, 0.1))
    # Peak should be shifted toward (0.25, 0.75) in physical coordinates
    centers = m.centers
    x_ctr, y_ctr = centers[0], centers[1]
    dist_from_peak = (x_ctr - 0.25) ** 2 + (y_ctr - 0.75) ** 2
    peak_idx = np.unravel_index(np.argmin(dist_from_peak), dist_from_peak.shape)
    assert f.values[peak_idx] > f.values.mean()


def test_field_sinusoidal():
    m = mesh_2d(nx=10, ny=20)
    f = field_sinusoidal(m, frequencies=(1.0, 2.0))
    assert f.shape == (10, 20)
    assert f.intensive is True
    # Values should oscillate
    assert f.values.min() < 0 and f.values.max() > 0


def test_field_sinusoidal_1d():
    m = mesh_1d(n_cells=100, length=2 * np.pi)
    f = field_sinusoidal(m, frequencies=(1.0,), phase=(0.0,))
    # sin(2π * 1 * x) at x=0.25 → sin(π/2) ≈ 1
    idx = int(0.25 / (2 * np.pi) * 100)
    assert np.isclose(f.values[idx], 1.0, atol=0.1)


def test_field_random_uniform():
    m = mesh_2d(nx=5, ny=5)
    f1 = field_random(m, seed=42)
    f2 = field_random(m, seed=42)
    assert f1.shape == (5, 5)
    assert_array_equal(f1.values, f2.values)  # same seed gives same values
    assert f1.values.min() >= 0 and f1.values.max() < 1


def test_field_random_normal():
    m = mesh_2d(nx=5, ny=5)
    f = field_random(m, seed=123, distribution="normal")
    assert f.shape == (5, 5)
    # Standard normal: mean ~0, std ~1
    assert abs(f.values.mean()) < 1.0
    assert 0.5 < f.values.std() < 1.5


def test_field_random_invalid_distribution():
    m = mesh_2d()
    with pytest.raises(ValueError, match="Unsupported distribution"):
        field_random(m, distribution="foobar")


# ─────────────────────────────────────────────────────────────────
# Fields container factory
# ─────────────────────────────────────────────────────────────────

def test_fields_collection_default():
    coll = fields_collection()
    assert isinstance(coll, Fields)
    assert coll.mesh.ndim == 2
    assert len(coll.data_names) == 4
    expected = {"constant", "gaussian", "sinusoidal", "random"}
    assert coll.data_names == expected


def test_fields_collection_custom_mesh():
    m = mesh_3d(nx=3, ny=4, nz=5)
    coll = fields_collection(mesh=m)
    assert isinstance(coll, Fields)
    assert coll.mesh is m
    assert all(coll[name].shape == (3, 4, 5) for name in coll.data_names)
