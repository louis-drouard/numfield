from numfield import CartesianField, Fields, merge_fields, CartesianMesh

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

import packaging

def test_field_mesh_consistency(coarse_field):
    assert coarse_field.values.shape == coarse_field.mesh.shape

def test_numpy_operation_on_field(coarse_field):
    
    values = coarse_field.values

    # numpy operation
    f_add2 = coarse_field + 2
    assert isinstance(f_add2, CartesianField) 
    assert_allclose(f_add2.values, values + 2)

    f_mul2 = coarse_field * 2
    assert isinstance(f_mul2, CartesianField) 
    assert_allclose(f_mul2.values, values * 2)

    f_div2 = coarse_field / 2
    assert isinstance(f_div2, CartesianField) 
    assert_allclose(f_div2.values, values / 2)

    f_sub2 = coarse_field - 2
    assert isinstance(f_sub2, CartesianField) 
    assert_allclose(f_sub2.values, values - 2)

    f_mod2 = coarse_field % 2
    assert isinstance(f_mod2, CartesianField) 
    assert_allclose(f_mod2.values, values % 2)

    f_ediv2 = coarse_field // 2
    assert isinstance(f_ediv2, CartesianField) 
    assert_allclose(f_ediv2.values, values // 2)

    # numpy inplace operation
    f_add2 += 2
    assert isinstance(f_add2, CartesianField) 
    assert_allclose(f_add2.values, values + 4)

    f_mul2 *= 2
    assert isinstance(f_mul2, CartesianField) 
    assert_allclose(f_mul2.values, values * 4)

    f_div2 /= 2
    assert isinstance(f_div2, CartesianField) 
    assert_allclose(f_div2.values, values / 4)

    f_sub2 -= 2
    assert isinstance(f_sub2, CartesianField) 
    assert_allclose(f_sub2.values, values - 4)

    # equalities as array
    assert_allclose(coarse_field, coarse_field.values)

def test_slicing(fine_field):
    plane_xy = fine_field[:,:,0]

    # check field and mesh dimension
    assert isinstance(plane_xy, CartesianField)
    assert isinstance(plane_xy.mesh, CartesianMesh)
    assert plane_xy.ndim == 2
    assert plane_xy.mesh.ndim == 2
    assert_allclose(plane_xy.mesh.axes[0], fine_field.mesh.axes[0])
    assert_allclose(plane_xy.mesh.axes[1], fine_field.mesh.axes[1])
    assert len(plane_xy.mesh.axes) == len(fine_field.mesh.axes) - 1

    # check values
    assert_allclose(plane_xy.values, fine_field.values[:,:,0])
    assert_allclose(fine_field[1,:,:], fine_field.values[1,:,:]) # plane yz
    assert_allclose(fine_field[:,2,:], fine_field.values[:,2,:]) # plan 2z

def test_numpy_functions(fine_field):
    fine_field[fine_field == 0] = 3.14 

    function_list = [
        np.mean, np.nanmean, np.median, np.nanmedian, np.sum, np.nansum, np.min, np.nanmin,
        np.max, np.nanmax, np.std, np.nanstd, np.var, np.nanvar, np.argmax, np.argmin,
        np.percentile, np.nanpercentile, np.quantile, np.nanquantile, np.ptp, np.median,
        np.average, np.prod, np.nanprod,
        np.zeros_like, np.where, np.round, np.rint, np.fix, np.floor, np.ceil, np.trunc, np.exp,
        np.log, np.log10, np.log2, np.exp2, np.expm1, np.log1p, np.logaddexp, np.logaddexp2,
        np.add, np.reciprocal, np.positive, np.negative, np.multiply, np.divide, np.power,
        np.subtract, np.floor_divide, np.fmod, np.mod, np.remainder, np.divmod,
        np.fmax, np.maximum, np.fmin, np.minimum, np.clip, np.sqrt, np.square, np.absolute,
        np.fabs, np.sign, np.heaviside, np.nan_to_num,
        np.ravel, np.unique, np.rot90, np.isnan, np.count_nonzero, np.cumsum, np.diff, np.ediff1d
        # np.transpose
        ]
    
    args_list = [
        [fine_field], [fine_field], [fine_field], [fine_field], [fine_field], [fine_field], [fine_field], [fine_field],
        [fine_field], [fine_field], [fine_field], [fine_field], [fine_field], [fine_field], [fine_field], [fine_field],
        [fine_field, 5], [fine_field, 5], [fine_field, 0.5], [fine_field, 0.5], [fine_field], [fine_field],
        [fine_field], [fine_field], [fine_field],
        [fine_field], [fine_field < 1, 2., 10.], [fine_field], [fine_field], [fine_field], [fine_field], [fine_field], [fine_field], [fine_field],
        [fine_field], [fine_field], [fine_field], [fine_field], [fine_field], [fine_field], [fine_field, fine_field], [fine_field, fine_field],
        [fine_field, fine_field], [fine_field], [fine_field], [fine_field], [fine_field, fine_field], [fine_field, fine_field], [fine_field, 2],
        [fine_field, 5.], [fine_field, 3.], [fine_field, 2], [fine_field, 2], [fine_field, 3.], [fine_field, 2.],
        [fine_field, fine_field*2], [fine_field, fine_field/2], [fine_field, fine_field/2], [fine_field, fine_field*2], [fine_field, None, 5], [fine_field], [fine_field], [fine_field],
        [fine_field], [fine_field], [fine_field, 0.5], [fine_field],
        [fine_field], [fine_field], [fine_field], [fine_field], [fine_field], [fine_field], [fine_field], [fine_field],
        # [fine_field]
    ]

    supported_to_array = {np.ravel, np.unique, np.rot90, np.savetxt, np.isnan, np.count_nonzero, np.cumsum, np.diff, np.ediff1d}
    if packaging.version.Version(np.__version__) >= packaging.version.Version("2.0"): # pyright: ignore[reportAttributeAccessIssue]
        function_list.extend([np.trapezoid, np.pow, np.gradient]) # pyright: ignore[reportAttributeAccessIssue]
        args_list.append([[fine_field], [fine_field, 2], [fine_field]])

    for func, arg in zip(function_list, args_list):
        # print(func)
        pure_numpy_result = func(*[a.values if isinstance(a, CartesianField) else a for a in arg])
        result = func(*arg) 
        assert_allclose(result, pure_numpy_result)

def test_numpy_method_on_field(fine_field):

    # sum
    summed_field = fine_field.sum(axis=1)
    assert summed_field.shape == (3, 2)
    assert_allclose(summed_field.values, fine_field.values.sum(axis=1))

    summed_field = fine_field.sum(axis=1, keepdims=True)
    assert summed_field.shape == (3, 1, 2)
    assert_allclose(summed_field.values, fine_field.values.sum(axis=1,  keepdims=True))

    # mean
    avrg_field = fine_field.mean(axis=1)
    assert avrg_field.shape == (3, 2)
    assert_allclose(avrg_field.values, fine_field.values.mean(axis=1))

    avrg_field = fine_field.mean(axis=1, keepdims=True)
    assert avrg_field.shape == (3, 1, 2)
    assert_allclose(avrg_field.values, fine_field.values.mean(axis=1,  keepdims=True))

    # std
    std_field = fine_field.std(axis=1)
    assert std_field.shape == (3, 2)
    assert_allclose(std_field.values, fine_field.values.std(axis=1))

    std_field = fine_field.std(axis=1, keepdims=True)
    assert std_field.shape == (3, 1, 2)
    assert_allclose(std_field.values, fine_field.values.std(axis=1,  keepdims=True))

    # var
    var_field = fine_field.var(axis=1)
    assert var_field.shape == (3, 2)
    assert_allclose(var_field.values, fine_field.values.var(axis=1))

    var_field = fine_field.var(axis=1, keepdims=True)
    assert var_field.shape == (3, 1, 2)
    assert_allclose(var_field.values, fine_field.values.var(axis=1,  keepdims=True))

    # min
    min_field = fine_field.min(axis=1)
    assert min_field.shape == (3, 2)
    assert_allclose(min_field.values, fine_field.values.min(axis=1))

    min_field = fine_field.min(axis=1, keepdims=True)
    assert min_field.shape == (3, 1, 2)
    assert_allclose(min_field.values, fine_field.values.min(axis=1,  keepdims=True))

    # max
    max_field = fine_field.max(axis=1)
    assert max_field.shape == (3, 2)
    assert_allclose(max_field.values, fine_field.values.max(axis=1))

    max_field = fine_field.max(axis=1, keepdims=True)
    assert max_field.shape == (3, 1, 2)
    assert_allclose(max_field.values, fine_field.values.max(axis=1,  keepdims=True))

    # ravel
    assert_allclose(fine_field.ravel(), fine_field.values.ravel())

    # flatten
    assert_allclose(fine_field.flatten(), fine_field.values.flatten())

def test_rot90(coarse_field):
    rot_field = coarse_field.rot90(1, (0,1))
    assert rot_field.mesh == coarse_field.mesh.rot90(1, (0,1))
    assert_allclose(rot_field.values, np.rot90(coarse_field, -1, (0,1)))
    assert_allclose(rot_field.values, np.rot90(coarse_field.values, -1, (0,1)))

def test_field_projection(fine_field, coarse_mesh):
    fine_to_coarse = fine_field.project_on(coarse_mesh)
    assert_allclose(fine_to_coarse, [[[0.5], [3.5/3]], [[5.5/3], [2.5]]])

    coarse_to_fine = fine_to_coarse.project_on(fine_field.mesh)
    assert_allclose(coarse_to_fine, [
        [[0.5, 0.5], [2.5/3, 2.5/3], [3.5/3, 3.5/3]],
        [[3.5/3, 3.5/3], [1.5, 1.5], [5.5/3, 5.5/3]],
        [[5.5/3, 5.5/3], [6.5/3, 6.5/3], [2.5, 2.5]]
        ])

    coarse_mesh_xy = coarse_mesh.drop_dimension(2)
    fine_to_coarse_xy = fine_field.project_on(coarse_mesh_xy)
    assert_allclose(fine_to_coarse_xy, [
        [[0.5, 0.5], [3.5/3, 3.5/3]],
        [[5.5/3, 5.5/3], [2.5, 2.5]]
        ])

def test_fields_construction(coarse_mesh, coarse_field, fine_field):
    fields = Fields(coarse_mesh)
    # check that all meshes are the same to unbias the following tests (who knows what fixture really do)
    assert id(coarse_mesh) == id(coarse_field.mesh) == id(fields.mesh)

    # test from_field and add_value
    fields = Fields.from_field(coarse_field)
    fields.add_values('test', coarse_field.values, False)
    assert 'test' in fields.fields
    assert 'coarse' in fields.fields

    # not the same mesh
    with pytest.raises(ValueError):
        fields.add_field(fine_field)
    coarse_field_deepcopy = coarse_field.copy()
    with pytest.raises(ValueError):
        fields.add_field(coarse_field_deepcopy)

    # Test __setitem__
    coarse_field_deepcopy.mesh = coarse_field.mesh
    coarse_field_deepcopy.name = "toto"
    fields['toto'] = coarse_field_deepcopy
    assert 'toto' in fields.fields

    # test data_names attribute
    assert fields.data_names == set(['coarse', 'toto', 'test'])

def test_categorical_field(coarse_field):
    fields = Fields.from_field(coarse_field)
    cat_values = [[['TOTO'], ['TITI']], [['TUTU'], ['TATA']]]
    cat_field = CartesianField('category_as_field', coarse_field.mesh, cat_values, intensive=True)
    fields.add_field(cat_field) # considered as a values field
    fields.add_values('category', cat_values) # considered as a categorical field

    # only 1 is considered a category
    assert set(fields.field_lookup.keys()) == {'category'}
    # the values of the categorical field are the index and 
    assert_allclose(fields.fields['category'], [[[2], [1]], [[3], [0]]])
    # careful with this test, as it can depends on the order (sorted may prove useful)
    assert_array_equal(fields.field_lookup['category'], ['TATA', 'TITI', 'TOTO', 'TUTU'])
    # but when accessed, both are equivalent
    assert_array_equal(fields.fields['category_as_field'], cat_values)
    assert_array_equal(fields['category'], cat_values)

def test_sum_by_category():
    # 10x10x5 Mesh
    dx = np.full(10, 1.0)   # 10 cells along x
    dy = np.full(10, 1.0)   # 10 cells along y
    dz = np.full(5,  2.0)   # 5  cells along z

    mesh = CartesianMesh(dx, dy, dz)
    
    Z = mesh.centers[2]   # center z-coordinates of the cells
    density_values = 1000 + 10 * Z   # kg/m^3
    density_field = CartesianField(
        name="density",
        mesh=mesh,
        values=density_values,
        intensive=True        # density → intensive
    )

    # Material field: water below z=5, steel above
    material = np.where(Z < 5, "water", "steel")

    fields = Fields(mesh)
    fields.add_field(density_field)
    fields.add_values("material", material)

    labels, means = fields.sum_by_category("density", "material")
    assert_allclose(means, [1070.00, 1020.00])

def test_merge_fields():
    dx = np.array([0.5, 0.5])
    dy = np.array([0.5, 0.5])
    mesh1 = CartesianMesh(dx, dy)    
    values = [[0., 1.], [2., 3.]]
    f1 = CartesianField('m1', mesh1, values, intensive=True)       
    mesh2 = CartesianMesh(dx, dy, origin=(1,1))    
    values = [[4., 5.], [6., 7.]]
    f2 = CartesianField('m2', mesh2, values, intensive=True)
    merged_field = merge_fields("merged", f1, f2)
    
    expected_values = [
        [0., 1., 0., 0.],
        [2., 3., 0., 0.],
        [0., 0., 4., 5.],
        [0., 0., 6., 7.]
        ]

    assert_allclose(merged_field.values, expected_values)
    assert_allclose(merged_field.mesh.axes[0], [0., 0.5, 1., 1.5, 2.])
    assert_allclose(merged_field.mesh.axes[1], [0., 0.5, 1., 1.5, 2.])

def test_combine_overlapping_extensive_field():
    dx = np.array([0.5, 0.5])
    dy = np.array([0.5, 0.5])
    mesh1 = CartesianMesh(dx, dy)    
    values = [[0., 1.], [2., 3.]]
    f1 = CartesianField('m1', mesh1, values, intensive=False)       
    mesh2 = CartesianMesh(dx, dy, origin=(0.5,0))    
    values = [[4., 5.], [6., 7.]]
    f2 = CartesianField('m2', mesh2, values, intensive=False)
    # overlapping cells are added (and normalized by cell volume to acccount for extensivity)
    combined_field = merge_fields("combined", f1, f2) 
    assert_allclose(combined_field.values, [[0., 1.], [6., 8.], [6., 7.]])
    
def test_combine_overlapping_intensive_field():
    dx = np.array([0.5, 0.5])
    dy = np.array([0.5, 0.5])
    mesh1 = CartesianMesh(dx, dy)    
    values = [[0., 1.], [2., 3.]]
    f1 = CartesianField('m1', mesh1, values, intensive=True)       
    mesh2 = CartesianMesh(dx, dy, origin=(0.5,0))    
    values = [[4., 5.], [6., 7.]]
    f2 = CartesianField('m2', mesh2, values, intensive=True)
    # overlapping cells are added (and normalized by cell volume to acccount for extensivity)
    combined_field = merge_fields("combined", f1, f2)
    assert_allclose(combined_field.values, [[0., 1.], [3., 4.],  [6., 7.]])

def test_field_normalize(fine_field):
    # Default normalization: sum should become 1
    normalized = fine_field.normalize()
    assert isinstance(normalized, CartesianField)
    assert normalized.shape == fine_field.shape
    assert normalized.intensive == fine_field.intensive
    assert_allclose(normalized.sum(), 1.0, atol=1e-10)

    # Normalize to a specific value (e.g., 5)
    normalized_5 = fine_field.normalize(norm=5.0)
    assert_allclose(normalized_5.sum(), 5.0, atol=1e-10)

    # Check that scaling is consistent: normalized_5 should be 5x normalized
    assert_allclose(normalized_5.values, 5 * normalized.values) 

def test_field_to_extensive(fine_field):
    extensive_values = [
        [[0.,         0.        ],
        [0.02777778, 0.02777778],
        [0.05555555, 0.05555555]],

        [[0.05555556, 0.05555556],
        [0.08333334, 0.08333334],
        [0.11111111, 0.11111111]],

        [[1/9, 1/9],
        [0.13888889, 0.13888889],
        [0.16666666, 0.16666666]],
    ]
    assert_allclose(fine_field.to_extensive(), extensive_values)

def test_field_to_intensive(fine_field):
    extensive_field = fine_field.to_extensive()
    assert_allclose(extensive_field.to_intensive(), fine_field)


# ─────────────────────────────────────────────────────────────────
# Transpose tests
# ─────────────────────────────────────────────────────────────────

def test_field_transpose_basic(coarse_mesh):
    from numfield.examples import field_constant
    f = field_constant(coarse_mesh, 1.0)
    ft = f.transpose()
    assert ft.shape == coarse_mesh.shape[::-1]
    assert ft.mesh.axes_names == coarse_mesh.axes_names[::-1]


def test_field_transpose_swap_2d():
    from numfield.examples import mesh_2d, field_gaussian
    m = mesh_2d()
    f = field_gaussian(m)
    ft = f.transpose((1, 0))
    assert ft.shape == (m.shape[1], m.shape[0])
    assert ft.mesh.axes_names == ["y", "x"]


def test_field_transpose_3d():
    from numfield.examples import mesh_3d, field_random
    m = mesh_3d(nx=2, ny=3, nz=4)
    f = field_random(m, seed=0)
    ft = f.transpose((2, 0, 1))
    assert ft.shape == (4, 2, 3)
    assert ft.mesh.axes_names == ["z", "x", "y"]


def test_mesh_transpose(coarse_mesh):
    mt = coarse_mesh.transpose((2, 1, 0))
    assert mt.shape == (coarse_mesh.shape[2], coarse_mesh.shape[1], coarse_mesh.shape[0])
    assert mt.axes_names == coarse_mesh.axes_names[::-1]


def test_field_transpose_invalid_axes(coarse_field):
    with pytest.raises(ValueError):
        coarse_field.transpose((0, 0))  # duplicate axes
    with pytest.raises(ValueError):
        coarse_field.transpose((0, 1))  # too few axes (3D field needs 3)


def test_mesh_transpose_invalid_axes(coarse_mesh):
    with pytest.raises(ValueError):
        coarse_mesh.transpose((0, 0))
    with pytest.raises(ValueError):
        coarse_mesh.transpose((0,))


# ─────────────────────────────────────────────────────────────────
# HDF5 round-trip tests
# ─────────────────────────────────────────────────────────────────

def test_mesh_hdf_roundtrip(tmp_path, coarse_mesh):
    filepath = tmp_path / "mesh.h5"
    coarse_mesh.to_hdf(str(filepath))
    m_loaded = CartesianMesh.from_hdf(str(filepath))
    assert coarse_mesh == m_loaded


def test_field_hdf_roundtrip(tmp_path, coarse_field):
    filepath = tmp_path / "field.h5"
    coarse_field.to_hdf(str(filepath))
    f_loaded = CartesianField.from_hdf(str(filepath))
    assert coarse_field == f_loaded


def test_fields_hdf_roundtrip(tmp_path, coarse_mesh):
    from numfield.examples import fields_collection
    coll = fields_collection(coarse_mesh)
    filepath = tmp_path / "fields.h5"
    coll.to_hdf(str(filepath))
    coll_loaded = Fields.from_hdf(str(filepath))
    assert coll.mesh == coll_loaded.mesh
    assert coll.data_names == coll_loaded.data_names