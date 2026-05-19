"""
Field package: Cartesian meshes, fields, and utilities.

This package provides:
- CartesianMesh: structured N-D Cartesian mesh
- CartesianField: field data defined on a Cartesian mesh
- Fields: container for multiple fields sharing a mesh
- merge_meshes / merge_fields: utilities for combining meshes/fields

For example meshes and fields useful for testing, import from `field.examples`:
    >>> from field.examples import mesh_2d, field_gaussian
"""

from .fields import Fields, CartesianField, merge_fields
from .mesh import CartesianMesh, merge_meshes

__all__ = [
    "Fields",
    "CartesianField",
    "CartesianMesh",
    "merge_fields",
    "merge_meshes",
]

# Optional: make examples available via `from field import mesh_2d` etc.
# These are exposed lazily to avoid polluting the core namespace.
import sys as _sys
from . import examples as _examples

for _name in _examples.__all__:
    globals()[_name] = getattr(_examples, _name)
    if _name not in __all__:
        __all__.append(_name) # pyright: ignore[reportUnsupportedDunderAll]

del _sys, _examples
