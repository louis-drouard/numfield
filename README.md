# numfield

A structured cartesian mesh field representation for scientific computing.

## Overview

`numfield` provides an intuitive framework for working with scalar fields defined on structured N-dimensional Cartesian meshes. It offers seamless integration with NumPy, allowing you to perform numerical operations while preserving mesh information.

## Features

- **N-dimensional Cartesian meshes**: Support for 1D, 2D, and 3D structured grids
- **NumPy integration**: Full compatibility with NumPy ufuncs and array operations
- **Intensive/extensive quantities**: Proper handling of different field types
- **HDF5 I/O**: Save and load fields and mesh data to/from HDF5 files
- **Visualization**: Built-in plotting methods for 1D, 2D, and 3D fields with interactive slicing
- **Field operations**: Projection, rotation, transposition, and merging of fields
- **Multi-field containers**: Manage multiple fields on a common mesh

## Installation

### From PyPI (recommended)

```bash
pip install numfield
```

### From source

```bash
git clone https://github.com/louis-drouard/numfield.git
cd numfield
pip install -e .
```

### Development installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from numfield import CartesianMesh, CartesianField, Fields

# Create a 2D mesh
mesh = CartesianMesh([1., 1.], [2., 2.], [5., 1., 4.])  # deltas (here x, y and z)

# Create a field on the mesh
values = np.random.rand(*mesh.shape)
field = CartesianField("temperature", mesh, values, intensive=True)

# Perform operations (preserves mesh information)
mean_temp = field.mean()
normalized = field.normalize()

# Plot the field
field.plot()

# Save to HDF5
field.to_hdf("output.h5")

# Load from HDF5
loaded_field = CartesianField.from_hdf("output.h5")
```

## Usage Examples

### Working with multiple fields

```python
from numfield import Fields

# Create a container for multiple fields
fields = Fields(mesh)
fields.add_values("temperature", np.random.rand(*mesh.shape))
fields.add_values("pressure", np.random.rand(*mesh.shape))

# Access individual fields
temp_field = fields["temperature"]
```

### Field projection and merging

```python
# Project a field onto a different mesh
target_mesh = CartesianMesh.from_linspace([0, 0], [2, 2], [20, 20])
projected_field = temp_field.project_on(target_mesh)

# Merge multiple fields from different regions
from numfield import merge_fields
combined_field = merge_fields("combined", field1, field2, field3)
```

### Interactive visualization

```python
# For 3D fields, use interactive slicing
field_3d.plot(axis=2, dynamic_colorbar=True, display_edges=False)
```

## API Reference

### Core Classes

- **`CartesianMesh`**: N-dimensional structured Cartesian mesh
- **`CartesianField`**: Scalar field defined on a Cartesian mesh
- **`Fields`**: Container for multiple fields sharing a mesh

### Key Operations

- **Arithmetic**: All NumPy operations (`+`, `-`, `*`, `/`, etc.)
- **Statistical**: `mean()`, `sum()`, `std()`, `min()`, `max()`, `describe()`
- **Transformation**: `project_on()`, `rot90()`, `transpose()`, `normalize()`
- **I/O**: `to_hdf()`, `from_hdf()`, `to_hdf_group()`, `from_hdf_group()`

## Requirements

- Python >= 3.10
- NumPy >= 1.26.4
- matplotlib
- h5py

## Development

### Running tests

```bash
pytest --mpl
```

The `--mpl` flag enables matplotlib baseline image comparison for visual regression testing.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This package was developed at the French Alternative Energies and Atomic Energy Commission (CEA).
