# File generated with docstub

from collections.abc import Hashable, Sequence

import numba
import numpy as np
import xarray
import xarray as xr
from _typeshed import Incomplete
from numpy.typing import ArrayLike

from . import _remove_indexes_to_reduce, sort

__all__ = ["histogram", "searchsorted", "ecdf"]

def hist_ufunc(data: Incomplete, bin_edges: Incomplete, res: Incomplete) -> None: ...
def histogram(
    da: xarray.DataArray,
    dims: Hashable | Sequence[Hashable],
    bins: ArrayLike | None = ...,
    density: bool = ...,
    **kwargs: Incomplete,
) -> xarray.DataArray: ...
def searchsorted_ufunc(da: Incomplete, v: Incomplete, res: Incomplete) -> None: ...
def searchsorted(
    da: xarray.DataArray,
    v: xarray.DataArray,
    dims: Hashable | Sequence[Hashable] | None = ...,
    **kwargs: Incomplete,
) -> None: ...
def ecdf(
    da: xarray.DataArray,
    dims: Hashable | Sequence[Hashable] | None = ...,
    *,
    npoints: int | None = ...,
    **kwargs: Incomplete,
) -> xarray.DataArray: ...
