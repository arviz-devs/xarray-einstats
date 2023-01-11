"""Stats, linear algebra and einops for xarray."""

from __future__ import annotations

import numpy as np
import xarray as xr

from .linalg import einsum, raw_einsum, einsum_path, matmul

__all__ = ["einsum", "raw_einsum", "einsum_path", "matmul", "zeros_ref", "ones_ref", "empty_ref"]

__version__ = "0.5.0.dev0"


def find_index(elem, to_search_in):
    for i, da in enumerate(to_search_in):
        if elem in da.dims:
            return (i, elem)
    raise ValueError(f"{elem} not found in any reference object")


def _create_ref(*args, dims, np_creator, dtype=None):
    if dtype is None and all(isinstance(arg, xr.DataArray) for arg in args):
        dtype = np.result_type(*[arg.dtype for arg in args])
    ref_idxs = [find_index(dim, args) for dim in dims]
    shape = [len(args[idx][dim]) for idx, dim in ref_idxs]
    # TODO: keep non indexing coords?
    coords = {dim: args[idx][dim] for idx, dim in ref_idxs if dim in args[idx].coords}
    return xr.DataArray(
        np_creator(shape, dtype=dtype),
        dims=dims,
        coords=coords,
    )


def zeros_ref(*args, dims, dtype=None):
    """Create a zeros DataArray from reference object(s).

    Creates a DataArray filled with zeros from reference
    DataArrays or Datasets and a list with the desired dimensions.

    Parameters
    ----------
    *args : iterable of DataArray or Dataset
        Reference objects from which the lengths and coordinate values (if any)
        of the given `dims` will be taken.
    dims : list of hashable
        List of dimensions of the output DataArray. Passed as is to the
        {class}`~xarray.DataArray` constructor.
    dtype : dtype, optional
        The dtype of the output array.
        If it is not provided it will be inferred from the reference
        DataArrays in `args` with :func:`numpy.result_type`.

    Returns
    -------
    DataArray

    See Also
    --------
    ones_ref, empty_ref
    """
    return _create_ref(*args, dims=dims, np_creator=np.zeros, dtype=dtype)


def empty_ref(*args, dims, dtype=None):
    """Create an empty DataArray from reference object(s).

    Creates an empty DataArray from reference
    DataArrays or Datasets and a list with the desired dimensions.

    Parameters
    ----------
    *args : iterable of DataArray or Dataset
        Reference objects from which the lengths and coordinate values (if any)
        of the given `dims` will be taken.
    dims : list of hashable
        List of dimensions of the output DataArray. Passed as is to the
        {class}`~xarray.DataArray` constructor.
    dtype : dtype, optional
        The dtype of the output array.
        If it is not provided it will be inferred from the reference
        DataArrays in `args` with :func:`numpy.result_type`.

    Returns
    -------
    DataArray

    See Also
    --------
    ones_ref, zeros_ref
    """
    return _create_ref(*args, dims=dims, np_creator=np.empty, dtype=dtype)


def ones_ref(*args, dims, dtype=None):
    """Create a ones DataArray from reference object(s).

    Creates a DataArray filled with ones from reference
    DataArrays or Datasets and a list with the desired dimensions.

    Parameters
    ----------
    *args : iterable of DataArray or Dataset
        Reference objects from which the lengths and coordinate values (if any)
        of the given `dims` will be taken.
    dims : list of hashable
        List of dimensions of the output DataArray. Passed as is to the
        {class}`~xarray.DataArray` constructor.
    dtype : dtype, optional
        The dtype of the output array.
        If it is not provided it will be inferred from the reference
        DataArrays in `args` with :func:`numpy.result_type`.

    Returns
    -------
    DataArray

    See Also
    --------
    empty_ref, zeros_ref
    """
    return _create_ref(*args, dims=dims, np_creator=np.ones, dtype=dtype)
