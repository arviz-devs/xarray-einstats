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
def _asdataarray(x_or_q, dim_name) -> None: ...
def _wrap_method(method: Incomplete) -> None: ...

class XrRV:
    def __init__(self, dist, *args, **kwargs) -> None: ...
    def _broadcast_args(self, args, kwargs) -> None: ...
    def rvs(
        self,
        *args,
        size=...,
        random_state=...,
        dims=...,
        coords=...,
        apply_kwargs=...,
        **kwargs,
    ) -> None: ...

class XrContinuousRV(XrRV):
    pass

class XrDiscreteRV(XrRV):
    pass

def _add_documented_method(cls, wrapped_cls, methods, extra_docs=...) -> None: ...

doc_extras: Incomplete
base_methods: Incomplete

class multivariate_normal:
    def __init__(self, mean=..., cov=..., dims=...) -> None: ...
    def _process_inputs(self, mean: Incomplete, cov: Incomplete, dims: Incomplete) -> None: ...
    def rvs(
        self, mean=..., cov=..., dims=..., *, size=..., rv_dims=..., random_state=...
    ) -> None: ...
    def logpdf(self, x, mean=..., cov=..., dims=...) -> None: ...
    def pdf(self, x, mean=..., cov=..., dims=...) -> None: ...

def _apply_nonreduce_func(func, da, dims, kwargs, func_kwargs=...) -> None: ...
def _apply_reduce_func(func, da, dims, kwargs, func_kwargs=...) -> None: ...
def rankdata(
    da: xarray.DataArray,
    dims: Hashable | Sequence[Hashable] | None = ...,
    *,
    method: str | None = ...,
    **kwargs: Incomplete,
) -> xarray.DataArray: ...
def gmean(da, dims=..., dtype=..., *, weights=..., **kwargs) -> None: ...
def hmean(da, dims=..., *, dtype=..., **kwargs) -> None: ...
def circmean(da, dims=..., *, high=..., low=..., nan_policy=..., **kwargs) -> None: ...
def circvar(da, dims=..., *, high=..., low=..., nan_policy=..., **kwargs) -> None: ...
def circstd(da, dims=..., *, high=..., low=..., nan_policy=..., **kwargs) -> None: ...
def kurtosis(da, dims=..., *, fisher=..., bias=..., nan_policy=..., **kwargs) -> None: ...
def skew(da, dims=..., *, bias=..., nan_policy=..., **kwargs) -> None: ...
def logsumexp(da, dims=..., *, b=..., return_sign=..., **kwargs) -> None: ...
def median_abs_deviation(
    da, dims=..., *, center=..., scale=..., nan_policy=..., **kwargs
) -> None: ...
