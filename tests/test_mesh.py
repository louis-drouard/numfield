import pytest
import numpy as np
from numpy.testing import assert_allclose
from numfield import CartesianMesh, merge_meshes

def test_attributes(fine_mesh):
    assert fine_mesh.ndim == 3
    assert_allclose(fine_mesh.axes[0], [0., 1/3, 2/3, 1.])
    assert_allclose(fine_mesh.axes[1], [0., 1/3, 2/3, 1.])
    assert_allclose(fine_mesh.axes[2], [0., 0.5, 1.])
    assert_allclose(fine_mesh.deltas[0], [1/3, 1/3, 1/3])
    assert_allclose(fine_mesh.deltas[1], [1/3, 1/3, 1/3])
    assert_allclose(fine_mesh.deltas[2], [0.5, 0.5])
    assert_allclose(fine_mesh.x, [0., 1/3, 2/3, 1.])
    assert_allclose(fine_mesh.y, [0., 1/3, 2/3, 1.])
    assert_allclose(fine_mesh.z, [0., 0.5, 1.])
    assert_allclose(fine_mesh.dx, [1/3, 1/3, 1/3])
    assert_allclose(fine_mesh.dy, [1/3, 1/3, 1/3])
    assert_allclose(fine_mesh.dz, [0.5, 0.5])

    assert fine_mesh.shape == (3, 3, 2)
    assert_allclose(fine_mesh.origin,(0., 0., 0.))
    assert_allclose(fine_mesh.center, (0.5, 0.5, 0.5))
    assert_allclose(fine_mesh.bounding_box,((0., 1.), (0., 1.), (0., 1.)))
    assert_allclose(fine_mesh.volume, 1.)
    assert_allclose(fine_mesh.size,(1., 1., 1.))
    x_center = np.broadcast_to(np.array([1/6, 1/2, 5/6]).reshape(3, 1, 1), (3,3, 2))
    y_center =  np.tile(np.tile(np.array([1/6, 1/2, 5/6])[:, None], (1, 2)), (3, 1, 1))
    z_center = np.broadcast_to(np.array([0.25, 0.75]).reshape(1, 1, 2), (3, 3, 2))
    assert_allclose(fine_mesh.centers[0],x_center)
    assert_allclose(fine_mesh.centers[1],y_center)
    assert_allclose(fine_mesh.centers[2],z_center)
    fine_mesh.volumes
    assert_allclose(fine_mesh.volumes, [
        [[1/18, 1/18], [1/18, 1/18], [1/18, 1/18]],
        [[1/18, 1/18], [1/18, 1/18], [1/18, 1/18]],
        [[1/18, 1/18], [1/18, 1/18], [1/18, 1/18]]
        ])

def test_is_contained_in(fine_mesh):
    bigger_mesh = CartesianMesh(*[d+1 for d in fine_mesh.deltas], origin=fine_mesh.origin)
    assert fine_mesh.is_contained_in(bigger_mesh)
    assert not bigger_mesh.is_contained_in(fine_mesh)

def test_is_submesh_of(coarse_mesh, fine_mesh):
    assert not coarse_mesh.is_submesh_of(fine_mesh)
    
    dx = [0.25, 0.25, 0.25, 0.25]
    dy = [0.25, 0.25, 0.25, 0.25]
    dz = [0.5, 0.5]
    mesh = CartesianMesh(dx, dy, dz)
    assert mesh.is_submesh_of(coarse_mesh)

def test_is_overlapping():
    mesh1 = CartesianMesh([0.5, 0.5], [0.5, 0.5], origin=(0, 0))

    mesh2 = CartesianMesh([0.5, 0.5], [0.5, 0.5], origin=(1, 1))
    assert not mesh1.is_overlapping(mesh2)

    mesh2 = CartesianMesh([0.5, 0.5], [0.5, 0.5], origin=(0, 1))
    assert not mesh1.is_overlapping(mesh2)

    mesh2 = CartesianMesh([0.5, 0.5], [0.5, 0.5], origin=(1, 0))
    assert not mesh1.is_overlapping(mesh2)

    mesh2 = CartesianMesh([0.5, 0.5], [0.5, 0.5], origin=(-1, -1))
    assert not mesh1.is_overlapping(mesh2)

    mesh2= CartesianMesh([0.5, 0.5], [0.5, 0.5], origin=(0.5, 0))
    assert mesh1.is_overlapping(mesh2)
  
    mesh2= CartesianMesh([0.5, 0.5], [0.5, 0.5], origin=(0, 0.5))
    assert mesh1.is_overlapping(mesh2)

    mesh2= CartesianMesh([0.5, 0.5], [0.5, 0.5], origin=(-0.5, 0.))
    assert mesh1.is_overlapping(mesh2)

    mesh2= CartesianMesh([0.5, 0.5], [0.5, 0.5], origin=(0, -0.5))
    assert mesh1.is_overlapping(mesh2)

def test_rot90():
    x = np.array([0., 0.3, 0.4, 0.6, 1.])
    y = np.array([0., 0.2, 0.5, 0.6, 1. ])
    z = np.array([0., 0.5, 0.7, 1.])
    disym_mesh = CartesianMesh.from_axes(x, y, z)    

    assert disym_mesh == disym_mesh.rot90(0, (0,1))
    assert disym_mesh == disym_mesh.rot90(4, (0,1))
    rot90 = disym_mesh.rot90(1, (0,1))
    rot180 = disym_mesh.rot90(2, (0,1))
    rot270 = disym_mesh.rot90(3, (0,1))

    assert CartesianMesh.from_axes([0., 0.4, 0.5, 0.8, 1.], [0., 0.3, 0.4, 0.6, 1.], [0., 0.5, 0.7, 1.]) == rot90
    assert CartesianMesh.from_axes([0., 0.4, 0.6, 0.7,  1. ], [0., 0.4, 0.5, 0.8,  1. ], [0., 0.5, 0.7, 1.]) == rot180
    assert CartesianMesh.from_axes([0., 0.2, 0.5, 0.6,  1. ], [0., 0.4, 0.6, 0.7,  1. ], [0., 0.5, 0.7, 1.]) == rot270

def test_drop_dimension(coarse_mesh, fine_mesh):
    coarse_2d = coarse_mesh.drop_dimension(1)
    assert isinstance(coarse_2d, CartesianMesh)

    fine_2d = fine_mesh.drop_dimension(0)
    assert isinstance(fine_2d, CartesianMesh) #returns mesh eventhough fine_mesh is a mesh3D

def test_projected_on():
    mesh2d = CartesianMesh([1.0, 1.0], [2.0, 3.0], origin=(0.0, 0.0))
    mesh1d = CartesianMesh([1.0], origin=(0.0,))

    # Broadcasting 1D → 2D
    mesh_broadcasted = mesh2d.projected_on(mesh1d)

    assert_allclose(mesh_broadcasted.axes[0], mesh1d.axes[0])
    assert_allclose(mesh_broadcasted.axes[1], mesh2d.axes[1])

def test_merge_meshes():

    dx = np.array([0.5, 0.5])
    dy = np.array([0.5, 0.5])
    mesh1 = CartesianMesh(dx, dy)    
    mesh2 = CartesianMesh(dx, dy, origin=(1,1))    

    new_mesh = merge_meshes(mesh1, mesh2)
    assert_allclose(new_mesh.axes[0], [0., 0.5, 1., 1.5, 2.])
    assert_allclose(new_mesh.axes[1], [0., 0.5, 1., 1.5, 2.])

    with pytest.raises(ValueError):
        merge_meshes()

    with pytest.raises(ValueError):
        merge_meshes(mesh1, mesh2.drop_dimension(1))

def test_mesh_linspace():
    mesh = CartesianMesh.from_linspace([1, 5], [2, 10], [3, 5], axes_names=['a', 'b'])
    assert mesh.is_regular()

    assert_allclose(mesh.axes[0], [1., 1.5, 2.])
    assert_allclose(mesh.axes[1], [5., 6.25, 7.5, 8.75, 10.])
    assert mesh.axes_names == ['a', 'b']