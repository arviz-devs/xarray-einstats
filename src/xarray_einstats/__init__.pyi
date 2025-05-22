# File generated with docstub

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence

import numpy as np
import xarray
import xarray as xr
from _typeshed import Incomplete

from .accessors import EinopsAccessor, LinAlgAccessor
from .linalg import einsum, einsum_path, matmul

__all__ = [
    "einsum",
    "einsum_path",
    "matmul",
    "zeros_ref",
    "ones_ref",
    "empty_ref",
    "LinAlgAccessor",
    "EinopsAccessor",
]

__version__: Incomplete

def sort(
    da: xarray.DataArray,
    dim: Hashable,
    kind: str | None = ...,
    stable: bool | None = ...,
    **kwargs: Incomplete,
) -> xarray.DataArray: ...
def _remove_indexes_to_reduce(da, dims) -> None: ...
def _find_index(elem: Incomplete, to_search_in: Incomplete) -> None: ...
def _create_ref(
    *args: Incomplete, dims: Incomplete, np_creator: Incomplete, dtype: Incomplete = ...
) -> None: ...
def zeros_ref(
    *args: Iterable[xarray.DataArray | xarray.Dataset],
    dims: Sequence[Hashable],
    dtype: np.typing.DTypeLike | None = ...,
) -> xarray.DataArray: ...
def empty_ref(
    *args: Iterable[xarray.DataArray | xarray.Dataset],
    dims: Sequence[Hashable],
    dtype: np.typing.DTypeLike | None = ...,
) -> xarray.DataArray: ...
def ones_ref(
    *args: Iterable[xarray.DataArray | xarray.Dataset],
    dims: Sequence[Hashable],
    dtype: np.typing.DTypeLike | None = ...,
) -> xarray.DataArray: ...
