# tests/conftest.py
import pytest
import numpy as np
from numfield import CartesianField, CartesianMesh

import matplotlib
matplotlib.use('Agg')

@pytest.fixture
def coarse_mesh():
    dx = np.array([0.5, 0.5])
    dy = np.array([0.5, 0.5])
    dz = np.array([1.])
    mesh = CartesianMesh(dx, dy, dz)    
    return mesh

@pytest.fixture
def fine_mesh():
    x = np.linspace(0., 1., 4)
    y = np.linspace(0., 1., 4)
    z = np.linspace(0., 1., 3)
    mesh = CartesianMesh.from_axes(x, y, z)    
    return mesh

@pytest.fixture
def coarse_field(coarse_mesh):
    values = [[[0.], [1.]], [[2.], [3.]]]
    coarse_field = CartesianField('coarse', coarse_mesh, values, intensive=True)       
    return coarse_field

@pytest.fixture
def fine_field(fine_mesh):
    values = [
        [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
        [[1.0, 1.0], [1.5, 1.5], [2.0, 2.0]],
        [[2.0, 2.0], [2.5, 2.5], [3.0, 3.0]]
        ]
    fine_field = CartesianField('fine', fine_mesh, values, intensive=True)       
    return fine_field