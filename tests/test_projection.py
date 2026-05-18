import pytest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from numfield.projection import build_1D_transfer_matrix


def test_projection_1D():
    src_mesh  = np.array([0.0, 0.5])
    target_mesh = np.array([0.0, 0.25, 0.5])
    src_values = np.array([1.0])

    # Intensive (e.g. density)
    T_int = build_1D_transfer_matrix(src_mesh, target_mesh, intensive=True)
    assert_allclose(np.tensordot(src_values, T_int, axes=(0,0)), [1.0, 1.0])

    # Extensive (e.g. total mass)
    T_ext = build_1D_transfer_matrix(src_mesh, target_mesh, intensive=False)
    assert_allclose(np.tensordot(src_values, T_ext, axes=(0,0)), [0.5, 0.5])
