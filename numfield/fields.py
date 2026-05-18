# local imports
from .mesh import CartesianMesh, merge_meshes
from .plotting import _get_norm
from .projection import project_ND
from .utils import slice_to_str

# external imports
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import h5py

# standard imports
from copy import deepcopy
import logging
import packaging

logger = logging.getLogger(__name__)

FLOATING_FIELD_PRECISION = 8
ATOL_FLOATING_FIELD_PRECISION = 10**(-FLOATING_FIELD_PRECISION)

class CartesianField(np.lib.mixins.NDArrayOperatorsMixin):
    """
    Scalar field defined on a structured Cartesian mesh.

    The field holds numerical or categorical data associated with each mesh cell
    and supports NumPy-like operations while preserving mesh information.

    Parameters
    ----------
    name : str
        Name of the field.
    mesh : CartesianMesh
        Underlying mesh on which the field is defined.
    values : array-like
        Field values. Shape must match the mesh cell shape (`mesh.shape`).
    intensive : bool
        Whether the field represents an intensive quantity
        (e.g., density, temperature) or an extensive one (e.g., mass, volume).

    Raises
    ------
    TypeError
        If `mesh` is not an instance of `CartesianMesh`.
    ValueError
        If `values.shape` does not match the mesh cell shape.

    Examples
    --------
    >>> from field.mesh import CartesianMesh
    >>> mesh = CartesianMesh([1, 1, 1], [2, 2, 2])
    >>> field = CartesianField("temperature", mesh, np.ones(mesh.shape), intensive=True)
    >>> field.sum()
    9.0
    """
    def __init__(self, name: str, mesh:CartesianMesh, values, intensive:bool) -> None:
        """
        Initialize a CartesianField instance.

        Parameters
        ----------
        name : str
            Name of the field.
        mesh : CartesianMesh
            Underlying mesh on which the field is defined.
        values : array-like
            Field values. Shape must match the mesh cell shape (`mesh.shape`).
        intensive : bool
            Whether the field represents an intensive quantity
            (e.g., density, temperature) or an extensive one (e.g., mass, volume).

        Raises
        ------
        TypeError
            If `mesh` is not an instance of `CartesianMesh`.
        ValueError
            If `values.shape` does not match the mesh cell shape.
        """
        if not isinstance(mesh ,CartesianMesh):
            raise TypeError(f"mesh argument must be of type {CartesianMesh} not {type(mesh)}")

        values = np.asarray(values)

        # Check that the field shape matches the number of cells (not nodes)
        if values.shape != mesh.shape:
            raise ValueError(f"Field shape {values.shape} does not match mesh shape {mesh.shape} for cells.")
        
        self.name = name
        self.mesh = mesh
        self.values = values
        self.intensive = intensive

    @classmethod
    def volumes(cls, mesh, name='volumes')->'CartesianField':
        ''' Create a field where its values are the volume of the mesh '''
        return cls(name, mesh, mesh.volumes, False)

    ### Export and import to/from hdf format
    def to_hdf_group(self, parent_group):
        """Write this field into an existing HDF5 group."""
        # Create group for the field
        grp = parent_group.create_group(self.name)
        grp.attrs["name"] = self.name
        grp.attrs["intensive"] = self.intensive
        grp.create_dataset("values", data=self.values)

        # Nested mesh group
        mesh_grp = grp.create_group("mesh")
        self.mesh.to_hdf_group(mesh_grp)

    @classmethod
    def from_hdf_group(cls, group):
        """Rebuild a CartesianField from an HDF5 group."""
        name = group.attrs["name"]
        intensive = bool(group.attrs["intensive"])
        values = group["values"][:]
        mesh = CartesianMesh.from_hdf_group(group["mesh"])
        return cls(name, mesh, values, intensive)

    def to_hdf(self, filename):
        """Save this field to a standalone HDF5 file."""
        with h5py.File(filename, "w") as f: # pyright: ignore[reportAttributeAccessIssue]
            self.to_hdf_group(f)

    @classmethod
    def from_hdf(cls, filename):
        """Load a CartesianField from an HDF5 file."""
        with h5py.File(filename, "r") as f: # pyright: ignore[reportAttributeAccessIssue]
            # we assume the field group is the first and only one
            group_name = next(iter(f.keys()))
            return cls.from_hdf_group(f[group_name])

    # --------------------------
    # NumPy protocol integration
    # --------------------------
    # -------- numpy cast --------
    def __array__(self, dtype=None, copy=False):
        """Return a NumPy array view of the field values."""
        return np.array(self.values, dtype=dtype, copy=copy)

    # -------- numpy operators --------
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Handle NumPy ufuncs with mesh compatibility checks.

        Returns
        -------
        CartesianField or ndarray
            A new field with the operation result, or ndarray if the operation
            breaks field structure (e.g., reduction to scalar).
        """
        logger.debug(f'calling {ufunc} ufunc on {self}')
        unwrapped_inputs = []
        for x in inputs:
            if isinstance(x, CartesianField):
                unwrapped_inputs.append(x.values)
                # Vérification compatibilité mesh/intensive
                if (x.mesh != self.mesh) or (x.intensive != self.intensive):
                    raise ValueError("Incompatible Field3D: mesh or intensive mismatch.")
            else:
                unwrapped_inputs.append(x)

        result = getattr(ufunc, method)(*unwrapped_inputs, **kwargs)

        if isinstance(result, tuple):
            return tuple(
                CartesianField(self.name, self.mesh, x, self.intensive) if isinstance(x, np.ndarray) else x
                for x in result
            )
        elif isinstance(result, np.ndarray):
            return CartesianField(self.name, self.mesh, result, self.intensive)
        else:
            return result

    # -------- numpy functions --------
    def __array_function__(self, func, types, args, kwargs):
        """
        Handle high-level NumPy functions.

        Supported functions:
            to CartesianField
                mean, nanmean, median, nanmedian, sum, nansum,min, nanmin,
                max, nanmax, std, nanstd, var, nanvar, argmax, argmin, 
                percentile, nanpercentile, quantile, nanquantile, ptp, median,
                average, prod, nanprod,
                zeros_like, where, round, rint, fix, floor, ceil, trunc, exp,
                log, log10, log2, exp2, expm1, log1p, logaddexp, logaddexp2,
                add, reciprocal, positive, negative, multiply, divide, power,
                pow, subtract, floor_divide, fmod, mod, remainder, divmod,
                fmax, maximum, fmin, minimum, clip, sqrt, square, absolute,
                fabs, sign, heaviside, nan_to_num, gradient
            
            To array/other output
                ravel, unique, rot90, savetxt, isnan, count_nonzero, cross, cumsum, diff, ediff1d, trapezoid

        """
        if not all(issubclass(t, CartesianField) for t in types):
            return NotImplemented

        # Supported numpy functions
        supported_field3d_change_dim = {
            np.mean, np.nanmean, np.median, np.nanmedian, np.sum, np.nansum, np.min, np.nanmin, np.max, np.nanmax,
            np.std, np.nanstd, np.var, np.nanvar, np.argmax, np.argmin, np.percentile, np.nanpercentile,
            np.quantile, np.nanquantile, np.ptp, np.median, np.average, np.prod, np.nanprod
            }
        supported_field3d = {
            np.zeros_like, np.where, np.round, np.rint, np.fix, np.floor, np.ceil, np.trunc, np.exp, np.log,
            np.log10, np.log2, np.exp2, np.expm1, np.log1p, np.logaddexp, np.logaddexp2, np.add, np.reciprocal,
            np.positive, np.negative, np.multiply, np.divide, np.power, np.subtract, np.floor_divide,
            np.fmod, np.mod, np.remainder, np.divmod, np.fmax, np.maximum, np.fmin, np.minimum, np.clip, np.sqrt,
            np.square, np.absolute, np.fabs, np.sign, np.heaviside, np.nan_to_num
            }
        supported_to_array = {np.ravel, np.unique, np.rot90, np.savetxt, np.isnan, np.count_nonzero, np.cumsum, np.diff, np.ediff1d}
        if packaging.version.Version(np.__version__) >= packaging.version.Version("2.0"): # pyright: ignore[reportAttributeAccessIssue]
            supported_field3d.add(np.pow) # pyright: ignore[reportAttributeAccessIssue]
            supported_field3d.add(np.gradient)
            supported_to_array.add(np.trapezoid) # pyright: ignore[reportAttributeAccessIssue]

        if func in supported_field3d_change_dim:
            values = [a.values if isinstance(a, CartesianField) else a for a in args]
            axis = kwargs.get('axis', None)
            keepdims = kwargs.get('keepdims', False)

            result = func(*values, **kwargs)
            # Handle scalar result
            if axis is None or (isinstance(axis, tuple) and len(axis) == self.ndim):
                return result  # pure scalar

            new_mesh = self.mesh.drop_dimension(axis, keepdims=keepdims) # type: ignore

            # Return a new field with reduced dimensions
            return CartesianField(self.name, new_mesh, result, self.intensive)
        
        elif func in supported_field3d:
            values = [a.values if isinstance(a, CartesianField) else a for a in args]
            result = func(*values, **kwargs)
            if isinstance(result, tuple):
                return tuple(CartesianField(func.__name__+f"_{i}", self.mesh.copy(), x, self.intensive) for i, x in enumerate(result))
            return CartesianField(func.__name__, self.mesh.copy(), result, self.intensive)
        
        elif func in supported_to_array:
            # Retourne un ndarray car la structure mesh ne colle plus
            values = [a.values if isinstance(a, CartesianField) else a for a in args]
            return func(*values, **kwargs)

        return NotImplemented

    # -------- numpy arrays properties --------
    def __getitem__(self, key):
        """Access field values by index."""

        result = self.values[key]

        # --- Case 1: Boolean or fancy indexing ---
        if isinstance(key, np.ndarray):
            if key.dtype == bool:
                # Mask selection
                return result
            else:
                return result
        
        # --- Case 2: Structured slicing (tuple, slice, int, etc.) ---
        # Normalize index to tuple
        if not isinstance(key, tuple):
            key = (key,)

        # Fill missing indices with full slices
        if len(key) < self.mesh.ndim:
            key = key + (slice(None),) * (self.mesh.ndim - len(key))

        new_axes = []
        for ax, sub_idx in zip(self.mesh.axes, np.atleast_1d(key)):
            if isinstance(sub_idx, (int, np.integer)):
                # Axis removed → skip it
                continue
            else:
                # keep axis
                if isinstance(sub_idx, slice):
                    start = sub_idx.start or 0
                    if start <0: # I do not understand why this step is necessary ( but it is)
                        start -= 1
                    stop = sub_idx.stop or len(ax)
                    if stop >0: # I do not understand why this step is necessary ( but it is)
                        stop += 1
                    step = sub_idx.step or 1
                    new_axes.append(ax[start:stop:step])
                else:
                    new_axes.append(ax[sub_idx])

        # If the result is an ndarray (not scalar), rebuild a field
        if isinstance(result, np.ndarray):
            new_mesh = CartesianMesh.from_axes(*new_axes)
            new_name = f"{self.name}@{slice_to_str(key)}"
            return CartesianField(new_name, new_mesh, result, self.intensive)
        
        return result
    
    def __setitem__(self, key, value):
        """Assign field values with compatibility checks."""
        if isinstance(value, CartesianField):
            if (value.mesh != self.mesh) or (value.intensive != self.intensive):
                raise ValueError("Incompatible Field3D assignment: mesh or intensive mismatch.")
            self.values[key] = value.values
        else:
            self.values[key] = value
    
    @property
    def dtype(self):
        """Data type of the field values."""
        return self.values.dtype

    @property
    def shape(self):
        """Shape of the field values (matches `mesh.shape`)."""
        return self.values.shape

    @property
    def ndim(self)->int:
        """Number of field dimensions."""
        return self.values.ndim

    @property
    def size(self)->int:
        """Size of field."""
        return self.values.size

    ### misc methods
    def ravel(self, *args, **kwargs):
        """Return the field as a flattened array (view)."""
        return np.ravel(self, *args, **kwargs)
        
    def flatten(self, *args, **kwargs):
        """Return a flattened copy of the field values."""
        return self.values.flatten(*args, **kwargs)
    
    ### field statistical description
    def sum(self, *args, **kwargs):
        """Return the sum of all field values."""
        if self.intensive:
            logger.warning('Summing intensive field values has ambiguous meaning, maybe you intend to apply summation on the values * volumes instead ?')
        return np.sum(self, *args, **kwargs)

    def mean(self, *args, **kwargs):
        """Return the mean of all field values."""
        return np.mean(self, *args, **kwargs)

    def std(self, *args, **kwargs):
        """Return the standard deviation of all field values.
        It behaves like numpy std (use ddof=1 to have the corrected deviation)"""
        return np.std(self, *args, **kwargs)

    def var(self, *args, **kwargs):
        """Return the variance of all field values."""
        return np.var(self, *args, **kwargs)

    def min(self, *args, **kwargs):
        """Return the min of all field values."""
        return np.min(self, *args, **kwargs)

    def max(self, *args, **kwargs):
        """Return the max of all field values."""
        return np.max(self, *args, **kwargs)

    def argmin(self, *args, **kwargs):
        """Return the min of all field values."""
        return np.min(self, *args, **kwargs)

    def argmax(self, *args, **kwargs):
        """Return the max of all field values."""
        return np.max(self, *args, **kwargs)

    def describe(self, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]):
        """
        Generate a descriptive statistical summary of the field values.

        Parameters
        ----------
        quantiles : list of float, optional
            Quantiles to compute, by default [0.05, 0.25, 0.5, 0.75, 0.95].

        Returns
        -------
        dict
            Dictionary containing 'size', 'nb_nan', 'mean', 'std', 'min',
            quantile values, and 'max' of the field values.
        """
        descriptive_stats = {
            'size': self.size,
            'nb_nan': int(np.isnan(self).sum()),
            'mean': float(np.nanmean(self)),
            'std': float(np.nanstd(self)),
            'min': float(np.nanmin(self))
            }
        for quantile in quantiles:
            descriptive_stats[f'{quantile:.3%}'] = float(np.nanquantile(self, quantile))

        descriptive_stats['max'] = float(np.nanmax(self))

        return descriptive_stats

    ### field comparison
    def __len__(self):
        """Number of elements along the first axis."""
        return len(self.values)
    
    def __iter__(self):
        """Iterator over field values."""
        return iter(self.values)
    
    def __eq__(self, other):
        """Check equality with another field or call numyp equality"""
        if isinstance(other, CartesianField):
            return (self.mesh == other.mesh 
                    and self.intensive is other.intensive
                    and np.allclose(self.values, other.values, atol=ATOL_FLOATING_FIELD_PRECISION))
        return self.values == other

    def __ne__(self, other):
        if isinstance(other, CartesianField):
            return (self.mesh != other.mesh 
                    or self.intensive is not other.intensive
                    or not np.allclose(self.values, other.values,atol=ATOL_FLOATING_FIELD_PRECISION))
        return self.values != other

    def __lt__(self, other):
        return self.values < other

    def __le__(self, other):
        return self.values <= other

    def __gt__(self, other):
        return self.values > other

    def __ge__(self, other):
        return self.values >= other
    
    ### field manipulation
    def _inplace_op(self ,other, op_name):
        """
        Perform an in-place operation on the field values.

        Parameters
        ----------
        other : CartesianField or scalar
            The operand to apply.
        op_name : str
            Name of the in-place operation method (e.g., '__iadd__').

        Returns
        -------
        CartesianField
            The modified field.
        """
        logging.debug(f"calling {getattr(self.values, op_name)}")
        if isinstance(other, CartesianField):
            if (self.mesh != other.mesh) or (self.intensive != other.intensive):
                raise ValueError(f'Imcompatible CartesianField for {op_name}')
            self.values = getattr(self.values, op_name)(other.values)
        else:
            self.values = getattr(self.values, op_name)(other)
        return self

    def __iadd__(self, other):  return self._inplace_op(other, '__iadd__')
    def __isub__(self, other):  return self._inplace_op(other, '__sub__')
    def __imul__(self, other):  return self._inplace_op(other, '__mul__')
    def __itruediv__(self, other):  return self._inplace_op(other, '__truediv__')

    def copy(self):
        """Return a deep copy of the field."""
        return deepcopy(self)

    def project_on(self, target_mesh: CartesianMesh, weights=None):
        """
        Project the field onto another mesh. If the target mesh has a 
        smaller dimension, it is automatically projected with
        `self.mesh.projected_on(target_mesh)`.

        Parameters
        ----------
        target_mesh : CartesianMesh
            Target mesh.

        Returns
        -------
        CartesianField
            New field defined on the target mesh.

        Raises
        ------
        ValueError
            If the field is non-numeric and the meshes are not nested.
        """
        if isinstance(target_mesh, CartesianField):
            target_mesh = target_mesh.mesh
        
        if (
            not np.issubdtype(self.dtype, np.number)
            and not self.mesh.is_submesh_of(target_mesh)
            and not target_mesh.is_submesh_of(self.mesh)
        ):
            raise ValueError(f'Projection is non-sensical because {self.name} is not numeric and one of both mesh is not a submesh of the other')
        
        # target mesh dimension is smaller than the current one
        if target_mesh.ndim < self.mesh.ndim:
            mesh_proj = self.mesh.projected_on(target_mesh)
        else:
            mesh_proj = target_mesh

        target_values = project_ND(
            self.mesh.axes,
            self.values,
            mesh_proj.axes,
            intensive=self.intensive,
            weights=weights
            )        
        target_field = CartesianField(self.name, mesh_proj, target_values, self.intensive)
        
        return target_field

    def rot90(self, nb_rotation: int, axes: tuple[int, int]= (0,1)):
        """
        Rotate both mesh and field values by multiples of 90°.

        Notes
        -----
        It is disctinct from np.rot90(self) which only return a ndarray where values have been rotated.
        Because of the way numpy index its 2D arrays top->bottom first and the left to right this method turn nb_rotation into -nb_rotation specifically for the call of np.rot90.
        applying np.rot90(self, k=1, (0,1)) won't yield the same result as self.rot90(k=1,(0,1)) but the same as self.rot90(k=-1,(0,1)) or self.rot90(k=1,(1,0))

        Parameters
        ----------
        nb_rotation : int
            Number of 90° rotations.
        axes : tuple of int
            Plane of rotation.

        Returns
        -------
        CartesianField
            Rotated field.
        """
        return self.__class__(
            f"",
            self.mesh.rot90(nb_rotation, axes),
            np.rot90(self.values, k=-nb_rotation, axes=axes), # numpy rotate in the anti-trigonometric direction (because of the way x-axis is top to bottom and y-axis is left to right in numpy)
            self.intensive
            )

    def transpose(self, axes=None):
        """
        Transpose both mesh and field values along specified axes.

        This method reorders the mesh axes and applies the same transposition
        to the field values using NumPy's transpose convention:
        the first axis becomes the last, the last becomes the first, and all
        intermediate axes remain in order.

        Parameters
        ----------
        axes : tuple of int, optional
            New ordering of the axes. Default is None, which reverses the axes order.
            For example, for a 3D field (axes: x, y, z):
            - `axes=None` or `axes=(2,1,0)` reverses axes → (z, y, x)
            - `axes=(1,0,2)` swaps first two axes → (y, x, z)
            - `axes=(0,2,1)` moves last axis to middle → (x, z, y)

        Returns
        -------
        CartesianField
            A new field with transposed mesh and values.

        Raises
        ------
        ValueError
            If `axes` contains invalid axis indices or duplicates.

        Examples
        --------
        >>> field_3d = CartesianField("temp", mesh_3d, values_3d, intensive=True)
        >>> field_3d_swapped = field_3d.transpose((1, 0, 2))  # swap x and y axes
        >>> field_3d_reversed = field_3d.transpose()  # reverse all axes
        """
        axes = tuple(reversed(range(self.ndim))) if axes is None else tuple(axes)

        if len(axes) != self.ndim:
            raise ValueError(f"transpose axes must match the field dimension {self.ndim}, got {len(axes)} axes")
        if len(set(axes)) != self.ndim:
            raise ValueError(f"transpose axes must be a permutation of 0..{self.ndim-1}, got {axes}")

        transposed_mesh = self.mesh.transpose(axes=axes)
        transposed_values = np.transpose(self.values, axes=axes)

        return self.__class__(self.name, transposed_mesh, transposed_values, self.intensive)

    def normalize(self, norm: float=1.0)->'CartesianField':
        '''
        Normalize the mesh so that the sum of its values is norm (default 1.) '''
        return norm * (self / self.sum())

    def to_extensive(self):
        """
        Convert an intensive field to extensive by multiplying with cell volumes.

        Returns
        -------
        CartesianField
            An extensive field where values are multiplied by cell volumes.
            If the field is already extensive, returns a copy of the field.
        """
        if self.intensive:
            extensive_field = self * self.mesh.volumes
            extensive_field.intensive = False
            return extensive_field
        return self.copy()

    def to_intensive(self):
        """
        Convert an extensive field to intensive by dividing by cell volumes.

        Returns
        -------
        CartesianField
            An intensive field where values are divided by cell volumes.
            If the field is already intensive, returns a copy of the field.
        """
        if self.intensive:
            return self.copy()
        intensive_field = self / self.mesh.volumes
        intensive_field.intensive = True
        return intensive_field

    ### representation methods
    def plot_1D(self, ax=None):
        """
        Plot the 1D field values against the mesh axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on. If None, creates new axes.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axis object.

        Raises
        ------
        ValueError
            If the field is not 1-dimensional.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if self.ndim != 1:
            raise ValueError(f'Plotting the field only work for 1D, not {self.ndim}D')

        plt.plot(self.mesh.axes[0][1:],self.values)
        if self.mesh.axes_names:
            ax.set_xlabel(self.mesh.axes_names[0])
        ax.set_ylabel(self.name)

        return fig, ax

    def plot(
            self,
            axis: int = 2,
            cbar_nb_levels: int | None = None,
            dynamic_colorbar: bool = False,
            grid: bool = False,
            show_tickslabels: bool|None = None,
            ax=None,
            cmap=None,
            vmin=None,
            vmax=None,
            vcenter=None,
            sym: bool = False,
            display_edges: bool = False,
            init_slice:int|None = None
        ):
        """
        Plot a 2D orthogonal slice of the field. This only work if the Field dimension is 2 or 3.

        Parameters
        ----------
        axis : int, default=2
            Axis orthogonal to the plotted plane (for 3D fields).
        dynamic_colorbar : bool, default=False
            If True, rescale colorbar when changing slices.
        grid : bool, default=False
            Show grid lines.
        show_tickslabels : bool|None, default=None
            Show axis tick labels if None,
            Show all mesh axis tick labels if True
            Show nothing if False
        cmap : colormap, optional
            Colormap for numerical fields.
        sym : bool, default=False
            Force symmetric color limits around v-center.
        display_edges : bool, default=False
            Display mesh cell edges.
        init_slice : int|None, default=None
            Initial slice to plot in the orthogonal axis (if 3D)
            if None, takes the middle slice

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axis object.
        """

        if self.ndim not in (1, 2, 3):
            raise NotImplementedError(f'Plotting the field only work for 2D or 3D Field, not {self.ndim}D')

        if self.ndim == 1:
            fig, ax = self.plot_1D(ax=ax)
            return fig, ax
        
        # Extract initial data slice
        if self.ndim == 2:
            plane_field = self
            max_slice_index = 0
            x_axis_name, y_axis_name = "x", "y"
        else:
            if axis==0:
                x_axis_name, y_axis_name, orthogonal_axis_name = "y", "z", "x"
            elif axis==1:
                x_axis_name, y_axis_name, orthogonal_axis_name = "x", "z", "y" #not direct (but humm)
            else:
                x_axis_name, y_axis_name, orthogonal_axis_name = "x", "y", "z"
                
            slice_number = (len(self.mesh.axes[axis])-1) // 2 if init_slice is None else init_slice
            max_slice_index = self.values.shape[axis] - 1
            plane_field = self[tuple(slice_number if i==axis else slice(None) for i in range(self.ndim))]
        data_slice = plane_field.values.T # pyright: ignore[reportAttributeAccessIssue]
        x,y = plane_field.mesh.axes # pyright: ignore[reportAttributeAccessIssue]

        #check if the values are categorical (not a number) to later adapt the colorbar accordingly
        values_are_categorical = False
        if not np.issubdtype(self.values.dtype, np.number):
            values_are_categorical = True

        # Set up figure and axis
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if cmap is None:
            cmap = matplotlib.colormaps["Spectral_r"] # pyright: ignore[reportAttributeAccessIssue]
        elif isinstance(cmap, str):
            cmap = plt.get_cmap(cmap) # pyright: ignore[reportAttributeAccessIssue]
        else:
            pass
        
        # Handle categorical fields
        if values_are_categorical:
            unique_values = np.unique(data_slice) if dynamic_colorbar else np.unique(self.values)
            value_map = {v: i for i, v in enumerate(unique_values)}
            data_slice = np.vectorize(value_map.get)(data_slice)
            cbar_nb_levels= len(unique_values)
            vmin0, vmax0 = 0, cbar_nb_levels # force the range of the colorbar
        
        else:
            # Compute color limits
            if dynamic_colorbar:
                vmin0 = np.nanmin(data_slice) if vmin is None else vmin
                vmax0 = np.nanmax(data_slice) if vmax is None else vmax
            else:
                vmin0 = np.nanmin(self.values) if vmin is None else vmin
                vmax0 = np.nanmax(self.values) if vmax is None else vmax
            if sym:
                vmax0 = max(abs(vmin0), abs(vmax0))
                vmin0 = -vmax0    
            if vmax0 == vmin0:
                vmin0 -= 0.5
                vmax0 += 0.5

        norm = _get_norm(vmin0, vmax0, vcenter, cbar_nb_levels, cmap)

        ### Initial pcolormesh
        pcmesh = ax.pcolormesh(
            x, y, data_slice, cmap=cmap, norm=norm,
            edgecolors="k", linewidth=0.5 if display_edges else 0
        )

        ### Create colorbar once
        cbar = fig.colorbar(pcmesh, ax=ax)
        fig.cbar = cbar # pyright: ignore[reportAttributeAccessIssue]
        cbar.set_label(self.name)
        # relocate tickslabel in the middle ofeach color and remove the ticks
        if values_are_categorical:
            cbar.set_ticks(np.arange(len(unique_values)) + 0.5, labels=unique_values) # type: ignore
            cbar.ax.tick_params(length=0)

        # Axis formatting
        ax.set_aspect("equal")
        ax.set_xlabel(x_axis_name)
        ax.set_ylabel(y_axis_name)
        if show_tickslabels is False:
            ax.set_xticks([])
            ax.set_yticks([])
        elif show_tickslabels is None:
            pass            
        else:
            ax.set_xticks(x)
            ax.set_yticks(y)

        if grid:
            ax.grid(color="black", linestyle="dashed", linewidth=0.7)
    
        ### add a slicer if there are more than one level in the axis orthogonal to the plotting plane
        if max_slice_index>0:
            axis_range = self.mesh.axes[axis][slice_number:slice_number+2] # type: ignore
            ax.set_title(f"{self.name} @ {orthogonal_axis_name}={slice_number} ({np.round(axis_range,4)})") # type: ignore
            plt.subplots_adjust(bottom=0.2)
            ax_slider = plt.axes((0.2, 0.05, 0.6, 0.04))
            slider = Slider(ax_slider, 'Slice', 0, max_slice_index, valinit=slice_number, valstep=1) # type: ignore
            fig.slider = slider # keep reference  # type: ignore 

            ### update the values in pcolormesh and the colorbar if requested/needed
            def update(val):
                slice_idx = int(slider.val)
                axis_range = self.mesh.axes[axis][slice_idx:slice_idx+2]
                data_slice_new = self[tuple(slice_idx if i==axis else slice(None) for i in range(self.ndim))].values.T # pyright: ignore[reportAttributeAccessIssue]
                if values_are_categorical:
                    unique_values = np.unique(data_slice_new) if dynamic_colorbar else np.unique(self.values)
                    value_map = {v: i for i, v in enumerate(unique_values)}
                    data_slice_new = np.vectorize(value_map.get)(data_slice_new)
                pcmesh.set_array(data_slice_new)
                                
                # Update colorbar limits dynamically
                if dynamic_colorbar:
                    if values_are_categorical:
                        vmin_new, vmax_new = 0, len(unique_values) # type: ignore
                        norm = _get_norm(vmin_new, vmax_new, vcenter, len(unique_values), cmap) # type: ignore
                    else:
                        # Compute color limits
                        vmin_new = np.nanmin(data_slice_new) if vmin is None else vmin
                        vmax_new = np.nanmax(data_slice_new) if vmax is None else vmax
                        if sym:
                            vmax_new = max(abs(vmin_new), abs(vmax_new))
                            vmin_new = -vmax_new  
                
                        norm = _get_norm(vmin_new, vmax_new, vcenter, cbar_nb_levels, cmap)
                    pcmesh.set_norm(norm)
                    cbar.update_normal(pcmesh)
                    if values_are_categorical:
                        cbar.set_ticks(np.arange(len(unique_values)) + 0.5, labels=unique_values) # type: ignore
                        cbar.ax.tick_params(length=0)

                ax.set_title(f"{self.name} @ {orthogonal_axis_name}={slice_idx} ({np.round(axis_range,4)})") # type: ignore
                fig.canvas.draw_idle()

            slider.on_changed(update)

        return fig, ax

    def __repr__(self):
        """String representation showing name and shape."""
        string = "intensive " if self.intensive else "extensive "
        string += f"{self.__class__.__name__}({self.name}, shape={self.values.shape})"
        return string
  

class Fields:
    """
        Container for multiple fields defined on a common Cartesian mesh.

        Attributes
        ----------
        mesh : CartesianMesh
            Underlying mesh.
        fields : dict of {str: CartesianField}
            Dictionary of field objects.
        field_lookup : dict of {str: ndarray}
            Lookup tables for categorical fields.

        Examples
        --------
        >>> from field.mesh import CartesianMesh
        >>> mesh = CartesianMesh([1, 1], [1, 1])
        >>> f = Fields(mesh)
        >>> f.add_values("temperature", np.ones(mesh.shape))
        >>> f["temperature"].mean()
        1.0
    """
    def __init__(self, mesh: CartesianMesh):
        self.mesh = mesh
        self.fields: dict[str, CartesianField] = {}
        self.field_lookup: dict[str, np.ndarray] = {}

    @classmethod
    def from_field(cls, field: CartesianField):
        """
        Create a Fields container from a single field.

        Parameters
        ----------
        field : CartesianField
            Field to include in the container.

        Returns
        -------
        Fields
            New container with the provided field.
        """
        fields = cls(field.mesh)
        fields[field.name] = field
        return fields

    @property
    def data_names(self)->set[str]:
        """Set of all field names in the container."""
        return set(self.fields.keys())

    @property
    def categorical_data_names(self) -> set[str]:
        """Set of names of categorical fields."""
        return set(self.field_lookup.keys())
    
    def __getitem__(self, name: str):
        """Access a field by name."""
        return self.get_field(name)
    
    def __setitem__(self, name: str, field: CartesianField):
        """Assign a field to the container."""
        if not isinstance(field, CartesianField):
            raise TypeError(f"item type is {type(field)} but only item of type {CartesianField} can be assigned directly , to assign array use set_field method instead")
        if field.mesh is not self.mesh:
            raise ValueError(f'Fields class already contains a mesh {id(self.mesh)}, all the fields it contains must has the same exact mesh object (received mesh {id(field.mesh)})')
        self.fields[name] = field       
        
    def __delitem__(self, name: str):
        """Delete a field by name."""
        return self.del_field(name)

    def get_field(self, name: str, decode=True)->CartesianField:
        """
        Retrieve a field by name.

        Parameters
        ----------
        name : str
            Name of the field.
        decode : bool, default=True
            Decode categorical fields if applicable.

        Returns
        -------
        CartesianField
            Requested field.
        """
        field = self.fields[name]
        if decode and name in self.field_lookup:
            return CartesianField(name, field.mesh, self.field_lookup[name][field.values], field.intensive)
        return field

    def add_field(self, field):
        """Add an existing CartesianField to the container."""
        self[field.name] = field

    def add_values(self, name, values, intensive=True):
        """ 
        Add a new field from raw values.

        Parameters
        ----------
        name : str
            Name of the field.
        values : array-like
            Field values.
        intensive : bool, default=True
            Whether the field is intensive.
        """
        values = np.asarray(values)
        if np.issubdtype(values.dtype, np.number):
            self.fields[name] = CartesianField(name, self.mesh, values, intensive)
        else:
            # unique_vals, indices = np.unique(values, return_inverse=True, sorted=False) # type: ignore:
            unique_vals, indices = np.unique(values, return_inverse=True) # type: ignore
            self.fields[name] = CartesianField(name, self.mesh, indices.reshape(values.shape), intensive)
            self.field_lookup[name] = unique_vals

    def del_field(self, name: str):
        """Remove a field and its lookup table if present."""
        self.fields.pop(name)
        if name in self.field_lookup:
            self.field_lookup.pop(name)

    ### retrieve additionnal field informations
    def sum_by_category(self, data_name, category_name):
        """
        Compute sums grouped by categories.

        Parameters
        ----------
        data_name : str
            Name of the numeric field to sum.
        category_name : str
            Name of the categorical field used for grouping.

        Returns
        -------
        categories : ndarray
            Unique categories.
        sums : ndarray
            Summed values per category.

        Raises
        ------
        KeyError
            If the category field is missing.
        """
        if category_name not in self.field_lookup:
            raise KeyError(f'{category_name} is not a category in {self}, available categorical data are {self.categorical_data_names}')

        indices = self.fields[category_name].values.ravel()
        values = self.fields[data_name].values
    
        if self.fields[data_name].intensive:
            volumes = self.mesh.volumes
            volumes_by_category = np.bincount(indices, weights=volumes.flatten())
            weights = (values * volumes).ravel()
            sum_by_category = np.bincount(indices, weights=weights) / volumes_by_category
        else:
            weights = values.ravel()           
            sum_by_category = np.bincount(indices, weights=weights)

        return self.field_lookup[category_name], sum_by_category
        return sum_by_category[self.fields[category_name]] # array filled with the sum
        
    ### field manipulation
    def rot90(self, nb_rotation: int, axes: tuple[int, int]= (0,1)):
        """
        Rotate the mesh and all contained fields.

        Parameters
        ----------
        nb_rotation : int
            Number of 90° rotations.
        axes : tuple of int
            Plane of rotation.
        """
        self.mesh = self.mesh.rot90(nb_rotation, axes)
        for name, field in self.fields.items():
            # numpy rotate in the anti-trigonometric direction, see Field.rot90 method for more explanation
            field.values = np.rot90(field.values, k=-nb_rotation, axes=axes)

    def to_hdf_group(self, parent_group):
        """
        Write the Fields container (mesh, fields, and lookup tables) into an existing HDF5 group.

        Parameters
        ----------
        parent_group : h5py.Group
            Parent HDF5 group in which to create the data.
        """
        # Save common mesh
        mesh_grp = parent_group.create_group("mesh")
        self.mesh.to_hdf_group(mesh_grp)

        # Save all fields (but not their mesh)
        fields_grp = parent_group.create_group("fields")
        for name, field in self.fields.items():
            grp = fields_grp.create_group(name)
            grp.attrs["name"] = field.name
            grp.attrs["intensive"] = field.intensive
            grp.create_dataset("values", data=field.values)

        # Optional categorical lookup
        if self.field_lookup:
            lookup_grp = parent_group.create_group("lookup")
            for k, v in self.field_lookup.items():
                if v.dtype.kind in {"U", "S"} or isinstance(v.flat[0], str):
                    # String array → use variable-length UTF-8 dtype
                    str_dtype = h5py.string_dtype(encoding="utf-8") # pyright: ignore[reportAttributeAccessIssue]
                    lookup_grp.create_dataset(k, data=np.array(v, dtype=object), dtype=str_dtype)
                else:
                    # Numeric array
                    lookup_grp.create_dataset(k, data=v)

    @classmethod
    def from_hdf_group(cls, group)->'Fields':
        """
        Rebuild a Fields object from an existing HDF5 group.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group containing the saved Fields data.

        Returns
        -------
        Fields
            The reconstructed Fields container.
        """
        mesh = CartesianMesh.from_hdf_group(group["mesh"])
        obj = cls(mesh)
        for name, grp_field in group["fields"].items():
            name = grp_field.attrs["name"]
            intensive = bool(grp_field.attrs["intensive"])
            values = grp_field["values"][:]
            field = CartesianField(name, mesh, values, intensive)
            obj.fields[name] = field

        if "lookup" in group:
            for k, ds in group["lookup"].items():
                # Ensure strings are read as np.str_ (UTF-8 decoded)
                if h5py.check_dtype(vlen=ds.dtype) is str: # pyright: ignore[reportAttributeAccessIssue]
                    obj.field_lookup[k] = np.array(ds[:], dtype=str)
                else:
                    obj.field_lookup[k] = ds[:]
        return obj

    def to_hdf(self, filename:str):
        """
        Save the Fields container to a standalone HDF5 file.

        Parameters
        ----------
        filename : str
            Path to the HDF5 file to be created.
        """
        import h5py
        with h5py.File(filename, "w") as f: # pyright: ignore[reportAttributeAccessIssue]
            self.to_hdf_group(f)

    @classmethod
    def from_hdf(cls, filename:str)->'Fields':
        """
        Load a Fields container from an HDF5 file.

        Parameters
        ----------
        filename : str
            Path to the HDF5 file to load.

        Returns
        -------
        Fields
            The loaded Fields container.
        """
        import h5py
        with h5py.File(filename, "r") as f: # pyright: ignore[reportAttributeAccessIssue]
            return cls.from_hdf_group(f)

    def __repr__(self):
        """String representation of the container."""
        return f"Fields(shape={self.mesh.shape}, fields={self.data_names})"


def merge_fields(name:str, *fields:CartesianField):
    """
    Combine multiple CartesianField objects into a single field.

    This function merges several CartesianField instances that share compatible
    dimensions and axes but occupy different spatial regions. It uses the CartesianField.project_on
    method to deal with both overlapping and non-overlapping fields.
    It merges their meshes into a unified global mesh, and 
    fills the corresponding regions of the resulting field with each field's data.

    Notes
    -----
    - Overlapping fields are projected using cells volumes as weights. A future implementation
      could add options to handle other given weights.
    - All fields must share the same `intensive` attribute. A future implementation
      could convert each field into the same intensive attribute (not sure this would be a desirable behaviour)

    Parameters
    ----------
    fields : list of CartesianField
        List of field objects to combine.

    Returns
    -------
    CartesianField
        A new CartesianField object defined on the merged mesh, containing the 
        values from all input fields placed at their correct positions.

    Raises
    ------
    NotImplementedError
        If any of the input fields if their `intensive` attributes differ.
    """

    global_mesh = merge_meshes(*[field.mesh for field in fields])

    intensive = fields[0].intensive
    if any(f.intensive is not intensive for f in fields):
        raise NotImplementedError(f"All the fields must have the same intensive attribute, first field is intensive={intensive}")

    # init to 0 the combined field
    combined_values = np.zeros(global_mesh.shape)
    weight_sum = np.zeros(global_mesh.shape) # for intensive
    for field in fields:
        projected_field = field.project_on(global_mesh)

        if intensive:
            #projected values are weighted by the volume of each cell and renormalized by total volume at the end
            projected_volumes = CartesianField.volumes(field.mesh).project_on(global_mesh)
            combined_values += projected_field.values * projected_volumes.values
            weight_sum += projected_volumes.values
        else:
            # direct addition for extenisve values
            combined_values += projected_field.values

    if intensive:
        # Avoid division per 0
        mask = weight_sum > 0
        combined_values[mask] /= weight_sum[mask]

    new_field = CartesianField(name, global_mesh, combined_values, intensive)

    return new_field

def compare_fields(field_ref: CartesianField, field_2: CartesianField, relative=True):
    """
    Compute the difference or relative difference between two fields.

    Parameters
    ----------
    field_ref : CartesianField
        Reference field for comparison.
    field_2 : CartesianField
        Field to compare against the reference.
    relative : bool, default=True
        If True, return the relative difference as a percentage:
        ``(field_2 - field_ref) / field_ref * 100``.
        If False, return the absolute difference: ``field_2 - field_ref``.

    Returns
    -------
    CartesianField
        A new field representing either the absolute or relative difference.

    Raises
    ------
    NotImplementedError
        If the two fields have different `intensive` attributes.
    """
    if field_ref.intensive is not field_2.intensive:
        raise NotImplementedError(f"Both fields must have the same intensive attribute, first field is intensive={field_ref.intensive}")
    
    diff = field_2.values - field_ref.values
    if relative:
        relative_gap = 100* np.divide(diff, field_ref.values, out=np.zeros_like(field_ref.values),where=field_ref.values!=0)
        return CartesianField(f"({field_2.name} - {field_ref.name}) / {field_ref.name} (%)", field_ref.mesh, relative_gap, field_ref.intensive)
    return CartesianField(f"({field_2.name} - {field_ref.name})", field_ref.mesh, diff, field_ref.intensive)

