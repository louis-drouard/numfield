import pytest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from numfield.projection import build_1D_transfer_matrix, project_ND


def test_projection_1D():
    src_mesh  = np.array([0.0, 0.5])
    target_mesh = np.array([0.0, 0.25, 0.5])
    src_values = np.array([1.0])

    # Intensive (e.g. density)
    T_int = build_1D_transfer_matrix(src_mesh, target_mesh, intensive=True)
    assert_allclose(np.tensordot(src_values, T_int, axes=(0,0)), [1.0, 1.0])
    result = project_ND((src_mesh,), src_values, (target_mesh,), intensive=True)
    assert_allclose(result, [1.0, 1.0])

    # Extensive (e.g. total mass)
    T_ext = build_1D_transfer_matrix(src_mesh, target_mesh, intensive=False)
    assert_allclose(np.tensordot(src_values, T_ext, axes=(0,0)), [0.5, 0.5])
    result = project_ND((src_mesh,), src_values, (target_mesh,), intensive=False)
    assert_allclose(result, [0.5, 0.5])

def test_projection_3D():
    x_src = np.array([0.0, 0.5, 1.0])
    y_src = np.array([0.0, 0.4, 0.8])
    z_src = np.array([0.0, 0.3, 0.6, 0.9])
    src_mesh = (x_src, y_src, z_src)

    x_trgt = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    y_trgt = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    z_trgt = np.array([0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9])
    target_mesh = (x_trgt, y_trgt, z_trgt)

    src_uniform = np.ones((2, 2, 3))
    result_uniform = project_ND(src_mesh, src_uniform, target_mesh, intensive=True)
    assert_allclose(result_uniform, 1.0)

    src_values = np.arange(1, 13).reshape(2, 2, 3)
    expected_intensive_result = [
        [[ 1.,  1.,  2.,  2.,  3.,  3.],
        [ 1.,  1.,  2.,  2.,  3.,  3.],
        [ 4.,  4.,  5.,  5.,  6.,  6.],
        [ 4.,  4.,  5.,  5.,  6.,  6.]],

        [[ 1.,  1.,  2.,  2.,  3.,  3.],
        [ 1.,  1.,  2.,  2.,  3.,  3.],
        [ 4.,  4.,  5.,  5.,  6.,  6.],
        [ 4.,  4.,  5.,  5.,  6.,  6.]],

        [[ 7.,  7.,  8.,  8.,  9.,  9.],
        [ 7.,  7.,  8.,  8.,  9.,  9.],
        [10., 10., 11., 11., 12., 12.],
        [10., 10., 11., 11., 12., 12.]],

        [[ 7.,  7.,  8.,  8.,  9.,  9.],
        [ 7.,  7.,  8.,  8.,  9.,  9.],
        [10., 10., 11., 11., 12., 12.],
        [10., 10., 11., 11., 12., 12.]]
        ]

    result_intensive = project_ND(src_mesh, src_values, target_mesh, intensive=True)
    assert_allclose(result_intensive, expected_intensive_result)

    expected_extensive_result = [
    [
        [0.125, 0.125, 0.25,  0.25,  0.375, 0.375],
        [0.125, 0.125, 0.25,  0.25,  0.375, 0.375],
        [0.5,   0.5,   0.625, 0.625, 0.75,  0.75 ],
        [0.5,   0.5,   0.625, 0.625, 0.75,  0.75 ]],

    [
        [0.125, 0.125, 0.25,  0.25,  0.375, 0.375],
        [0.125, 0.125, 0.25,  0.25,  0.375, 0.375],
        [0.5,   0.5,   0.625, 0.625, 0.75,  0.75 ],
        [0.5,   0.5,   0.625, 0.625, 0.75,  0.75 ]],

    [
        [0.875, 0.875, 1.,    1.,    1.125, 1.125],
        [0.875, 0.875, 1.,    1.,    1.125, 1.125],
        [1.25,  1.25,  1.375, 1.375, 1.5,   1.5  ],
        [1.25,  1.25,  1.375, 1.375, 1.5,   1.5  ]],

    [
        [0.875, 0.875, 1.,    1.,    1.125, 1.125],
        [0.875, 0.875, 1.,    1.,    1.125, 1.125],
        [1.25,  1.25,  1.375, 1.375, 1.5,   1.5  ],
        [1.25,  1.25,  1.375, 1.375, 1.5,   1.5  ]]
    ]
    result_extensive = project_ND(src_mesh, src_values, target_mesh, intensive=False)
    assert_allclose(result_extensive, expected_extensive_result)

def test_projection_3D_overlapping_nonuniform():
    """
    Test 3D projection with a simple (3,3,3) non-uniform overlapping case.
    
    Source mesh: 2 cells per dimension with non-uniform spacing
    Target mesh: 3 cells per dimension that overlap source cells
    
    This allows manual verification of the expected results.
    """
    # Simple non-uniform source mesh: 2 cells in each dimension
    x_src = np.array([0.0, 1.0, 3.0])  # cells: [0,1] width=1, [1,3] width=2
    y_src = np.array([0.0, 1.0, 3.0])
    z_src = np.array([0.0, 1.0, 3.0])
    src_mesh = (x_src, y_src, z_src)
    
    # Target mesh: 3 cells that overlap the source cells non-uniformly
    x_trgt = np.array([0.0, 0.5, 1.5, 3.0])  # cells: [0,0.5], [0.5,1.5], [1.5,3.0]
    y_trgt = np.array([0.0, 0.5, 1.5, 3.0])
    z_trgt = np.array([0.0, 0.5, 1.5, 3.0])
    target_mesh = (x_trgt, y_trgt, z_trgt)
    
    # Simple source values: each cell has a distinct value
    src_values = np.arange(1, 9).reshape(2, 2, 2)

    expected_extensive = [[
        [0.125,   0.25,    0.375  ],
        [0.3125,  0.5625,  0.75   ],
        [0.5625,  0.9375,  1.125  ]],
        [[0.4375,  0.75,    0.9375 ],
        [0.84375, 1.40625, 1.6875 ],
        [1.21875, 1.96875, 2.25   ]],
        [[0.9375,  1.5,     1.6875 ],
        [1.59375, 2.53125, 2.8125 ],
        [1.96875, 3.09375, 3.375  ]]]
    
    result_extensive = project_ND(src_mesh, src_values, target_mesh, intensive=False)
    assert_allclose(result_extensive, expected_extensive, rtol=1e-10)

    expected_intensive = [
        [[1.,  1.5, 2., ],
        [2.,  2.5, 3., ],
        [3.,  3.5, 4., ]],

        [[3.,  3.5, 4., ],
        [4.,  4.5, 5., ],
        [5.,  5.5, 6., ]],

        [[5.,  5.5, 6., ],
        [6.,  6.5, 7., ],
        [7.,  7.5, 8., ]]]
    
    result_intensive = project_ND(src_mesh, src_values, target_mesh, intensive=True)
    assert_allclose(result_intensive, expected_intensive, rtol=1e-10)