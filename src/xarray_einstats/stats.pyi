# File generated with docstub

from collections.abc import Hashable, Sequence

import numpy as np
import xarray
import xarray as xr
from _typeshed import Incomplete
from numpy.linalg import LinAlgError
from scipy import special, stats

from . import _remove_indexes_to_reduce
from .linalg import cholesky, eigh

__all__ = [
    "XrContinuousRV",
    "XrDiscreteRV",
    "multivariate_normal",
    "circmean",
    "circstd",
    "circvar",
    "gmean",
    "hmean",
    "kurtosis",
    "rankdata",
    "skew",
]

_SCIPY_RV_MAP: Incomplete

def get_default_dims(dims: list[str]) -> list[str]: ...
def _asdataarray(x_or_q: Incomplete, dim_name: Incomplete) -> None: ...
def _wrap_method(method: Incomplete) -> None: ...

class XrRV:
    def __init__(
        self, dist: Incomplete, *args: Incomplete, **kwargs: Incomplete
    ) -> None: ...
    def _broadcast_args(self, args: Incomplete, kwargs: Incomplete) -> None: ...
    def rvs(
        self,
        *args: Incomplete,
        size: Incomplete = ...,
        random_state: Incomplete = ...,
        dims: Incomplete = ...,
        coords: Incomplete = ...,
        apply_kwargs: Incomplete = ...,
        **kwargs: Incomplete,
    ) -> None: ...

class XrContinuousRV(XrRV):
    pass

class XrDiscreteRV(XrRV):
    pass

def _add_documented_method(
    cls: Incomplete,
    wrapped_cls: Incomplete,
    methods: Incomplete,
    extra_docs: Incomplete = ...,
) -> None: ...

doc_extras: Incomplete
base_methods: Incomplete

class multivariate_normal:
    def __init__(
        self, mean: Incomplete = ..., cov: Incomplete = ..., dims: Incomplete = ...
    ) -> None: ...
    def _process_inputs(
        self, mean: Incomplete, cov: Incomplete, dims: Incomplete
    ) -> None: ...
    def rvs(
        self,
        mean: Incomplete = ...,
        cov: Incomplete = ...,
        dims: Incomplete = ...,
        *,
        size: Incomplete = ...,
        rv_dims: Incomplete = ...,
        random_state: Incomplete = ...,
    ) -> None: ...
    def logpdf(
        self,
        x: Incomplete,
        mean: Incomplete = ...,
        cov: Incomplete = ...,
        dims: Incomplete = ...,
    ) -> None: ...
    def pdf(
        self,
        x: Incomplete,
        mean: Incomplete = ...,
        cov: Incomplete = ...,
        dims: Incomplete = ...,
    ) -> None: ...

def _apply_nonreduce_func(
    func: Incomplete,
    da: Incomplete,
    dims: Incomplete,
    kwargs: Incomplete,
    func_kwargs: Incomplete = ...,
) -> None: ...
def _apply_reduce_func(
    func: Incomplete,
    da: Incomplete,
    dims: Incomplete,
    kwargs: Incomplete,
    func_kwargs: Incomplete = ...,
) -> None: ...
def rankdata(
    da: xarray.DataArray,
    dims: Hashable | Sequence[Hashable] | None = ...,
    *,
    method: str | None = ...,
    **kwargs: Incomplete,
) -> xarray.DataArray: ...
def gmean(
    da: Incomplete,
    dims: Incomplete = ...,
    dtype: Incomplete = ...,
    *,
    weights: Incomplete = ...,
    **kwargs: Incomplete,
) -> None: ...
def hmean(
    da: Incomplete,
    dims: Incomplete = ...,
    *,
    dtype: Incomplete = ...,
    **kwargs: Incomplete,
) -> None: ...
def circmean(
    da: Incomplete,
    dims: Incomplete = ...,
    *,
    high: Incomplete = ...,
    low: Incomplete = ...,
    nan_policy: Incomplete = ...,
    **kwargs: Incomplete,
) -> None: ...
def circvar(
    da: Incomplete,
    dims: Incomplete = ...,
    *,
    high: Incomplete = ...,
    low: Incomplete = ...,
    nan_policy: Incomplete = ...,
    **kwargs: Incomplete,
) -> None: ...
def circstd(
    da: Incomplete,
    dims: Incomplete = ...,
    *,
    high: Incomplete = ...,
    low: Incomplete = ...,
    nan_policy: Incomplete = ...,
    **kwargs: Incomplete,
) -> None: ...
def kurtosis(
    da: Incomplete,
    dims: Incomplete = ...,
    *,
    fisher: Incomplete = ...,
    bias: Incomplete = ...,
    nan_policy: Incomplete = ...,
    **kwargs: Incomplete,
) -> None: ...
def skew(
    da: Incomplete,
    dims: Incomplete = ...,
    *,
    bias: Incomplete = ...,
    nan_policy: Incomplete = ...,
    **kwargs: Incomplete,
) -> None: ...
def logsumexp(
    da: Incomplete,
    dims: Incomplete = ...,
    *,
    b: Incomplete = ...,
    return_sign: Incomplete = ...,
    **kwargs: Incomplete,
) -> None: ...
def median_abs_deviation(
    da: Incomplete,
    dims: Incomplete = ...,
    *,
    center: Incomplete = ...,
    scale: Incomplete = ...,
    nan_policy: Incomplete = ...,
    **kwargs: Incomplete,
) -> None: ...
