# File generated with docstub

import numbers
import warnings
from collections.abc import Hashable, Sequence
from typing import Literal

import numpy as np
import xarray
import xarray as xr
from _typeshed import Incomplete
from numpy.typing import NDArray

__all__ = [
    "matrix_power",
    "matrix_transpose",
    "cholesky",
    "qr",
    "svd",
    "eig",
    "eigh",
    "eigvals",
    "eigvalsh",
    "norm",
    "cond",
    "det",
    "matrix_rank",
    "slogdet",
    "trace",
    "diagonal",
    "solve",
    "inv",
    "pinv",
]

class MissingMonkeypatchError(Exception):
    pass

def get_default_dims(da1_dims: list[str], d2_dims=...) -> list[str]: ...
def _attempt_default_dims(func, da1_dims, da2_dims=...) -> None: ...

class PairHandler:
    def __init__(self, all_dims: Incomplete, keep_dims: Incomplete) -> None: ...
    def process_dim_da_pair(self, da: Incomplete, dim_sublist: Incomplete) -> None: ...
    def get_out_subscript(self) -> None: ...

def _einsum_parent(
    dims: list[list[str]], *operands: xarray.DataArray, keep_dims: set = ...
) -> None: ...
def _translate_pattern_string(subscripts) -> None: ...
def _einsum_path(dims, *operands, keep_dims=..., optimize=..., **kwargs) -> None: ...
def einsum_path(
    dims: list[list[str]],
    *operands: xarray.DataArray,
    keep_dims=...,
    optimize: str | None = ...,
    **kwargs: dict,
) -> None: ...
def _einsum(
    dims, *operands, keep_dims=..., out_append=..., einsum_kwargs=..., **kwargs
) -> None: ...
def einsum(
    dims: str | list[list[str]],
    *operands: xarray.DataArray,
    keep_dims: set = ...,
    out_append: str = ...,
    einsum_kwargs: dict | None = ...,
    **kwargs: Incomplete,
) -> xarray.DataArray: ...
def matmul(
    da: xarray.DataArray,
    db: xarray.DataArray,
    dims: Sequence[Hashable] | None = ...,
    *,
    out_append: str = ...,
    **kwargs: Incomplete,
) -> xarray.DataArray: ...
def matrix_transpose(da: xarray.DataArray, dims: list[str]) -> xarray.DataArray: ...
def matrix_power(
    da: xarray.DataArray, n: int, dims: Sequence[Hashable] | None = ..., **kwargs: Incomplete
) -> xarray.DataArray: ...
def cholesky(
    da: xarray.DataArray, dims: Sequence[Hashable] | None = ..., **kwargs: Incomplete
) -> xarray.DataArray: ...
def qr(
    da: xarray.DataArray,
    dims: Sequence[Hashable] | None = ...,
    *,
    mode: str = ...,
    out_append: str = ...,
    **kwargs: Incomplete,
) -> tuple[xarray.DataArray, xarray.DataArray] | xarray.DataArray: ...
def svd(
    da: xarray.DataArray,
    dims: Sequence[Hashable] | None = ...,
    *,
    full_matrices: bool = ...,
    compute_uv=...,
    hermitian: bool = ...,
    out_append: str = ...,
    **kwargs: Incomplete,
) -> tuple[xarray.DataArray, xarray.DataArray, xarray.DataArray] | xarray.DataArray: ...
def eig(
    da: xarray.DataArray, dims: Sequence[Hashable] | None = ..., **kwargs: Incomplete
) -> tuple[xarray.DataArray, xarray.DataArray]: ...
def eigh(
    da: xarray.DataArray, dims: Sequence[Hashable] | None = ..., *, UPLO=..., **kwargs: Incomplete
) -> tuple[xarray.DataArray, xarray.DataArray]: ...
def eigvals(
    da: xarray.DataArray, dims: Sequence[Hashable] | None = ..., **kwargs: Incomplete
) -> xarray.DataArray: ...
def eigvalsh(
    da: xarray.DataArray, dims: Sequence[Hashable] | None = ..., *, UPLO=..., **kwargs: Incomplete
) -> xarray.DataArray: ...
def norm(
    da: xarray.DataArray,
    dims: Sequence[Hashable] | None = ...,
    *,
    ord: numbers.Number | Literal["fro", "nuc"] | None = ...,
    **kwargs: Incomplete,
) -> xarray.DataArray: ...
def cond(
    da: xarray.DataArray,
    dims: Sequence[Hashable] | None = ...,
    *,
    p: int | Literal[np.inf, "fro"] | None = ...,
    **kwargs: Incomplete,
) -> xarray.DataArray: ...
def det(
    da: xarray.DataArray, dims: Sequence[Hashable] | None = ..., **kwargs: Incomplete
) -> xarray.DataArray: ...
def matrix_rank(
    da: xarray.DataArray,
    dims: Sequence[Hashable] | None = ...,
    *,
    tol: float | xarray.DataArray | None = ...,
    rtol: float | xarray.DataArray | None = ...,
    hermitian: bool = ...,
    **kwargs: Incomplete,
) -> xarray.DataArray: ...
def slogdet(
    da: xarray.DataArray, dims: Sequence[Hashable] | None = ..., **kwargs: Incomplete
) -> tuple[xarray.DataArray, xarray.DataArray]: ...
def trace(
    da: xarray.DataArray,
    dims: Sequence[Hashable] | None = ...,
    *,
    offset: int = ...,
    dtype: np.typing.DTypeLike | None = ...,
    out: NDArray | None = ...,
    **kwargs,
) -> xarray.DataArray: ...
def diagonal(
    da: xarray.DataArray,
    dims: Sequence[Hashable] | None = ...,
    *,
    offset: int = ...,
    **kwargs: Incomplete,
) -> xarray.DataArray: ...
def solve(
    da: xarray.DataArray,
    db: xarray.DataArray,
    dims: Sequence[Hashable] | None = ...,
    **kwargs: Incomplete,
) -> xarray.DataArray: ...
def inv(
    da: xarray.DataArray, dims: Sequence[Hashable] | None = ..., **kwargs: Incomplete
) -> xarray.DataArray: ...
def pinv(
    da: xarray.DataArray,
    dims: Sequence[Hashable] | None = ...,
    *,
    rcond: float | xarray.DataArray | None = ...,
    rtol: float | xarray.DataArray | None = ...,
    hermitian: bool = ...,
    **kwargs: Incomplete,
) -> xarray.DataArray: ...
