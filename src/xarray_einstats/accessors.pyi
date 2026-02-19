# File generated with docstub

import xarray as xr
from _typeshed import Incomplete

from .linalg import (
    cholesky,
    cond,
    det,
    diagonal,
    eig,
    eigh,
    eigvals,
    eigvalsh,
    inv,
    matrix_power,
    matrix_rank,
    matrix_transpose,
    norm,
    qr,
    slogdet,
    solve,
    svd,
    trace,
)

class LinAlgAccessor:
    def __init__(self, xarray_obj: Incomplete) -> None: ...
    def matrix_transpose(self, dims: Incomplete) -> None: ...
    def matrix_power(
        self, n: Incomplete, dims: Incomplete = ..., **kwargs: Incomplete
    ) -> None: ...
    def cholesky(self, dims: Incomplete = ..., **kwargs: Incomplete) -> None: ...
    def qr(
        self,
        dims: Incomplete = ...,
        *,
        mode: Incomplete = ...,
        out_append: Incomplete = ...,
        **kwargs: Incomplete,
    ) -> None: ...
    def svd(
        self,
        dims: Incomplete = ...,
        *,
        full_matrices: Incomplete = ...,
        compute_uv: Incomplete = ...,
        hermitian: Incomplete = ...,
        out_append: Incomplete = ...,
        **kwargs: Incomplete,
    ) -> None: ...
    def eig(self, dims: Incomplete = ..., **kwargs: Incomplete) -> None: ...
    def eigh(
        self, dims: Incomplete = ..., *, UPLO: Incomplete = ..., **kwargs: Incomplete
    ) -> None: ...
    def eigvals(self, dims: Incomplete = ..., **kwargs: Incomplete) -> None: ...
    def eigvalsh(
        self, dims: Incomplete = ..., *, UPLO: Incomplete = ..., **kwargs: Incomplete
    ) -> None: ...
    def norm(
        self, dims: Incomplete = ..., *, ord: Incomplete = ..., **kwargs: Incomplete
    ) -> None: ...
    def cond(
        self, dims: Incomplete = ..., *, p: Incomplete = ..., **kwargs: Incomplete
    ) -> None: ...
    def det(self, dims: Incomplete = ..., **kwargs: Incomplete) -> None: ...
    def matrix_rank(
        self,
        dims: Incomplete = ...,
        *,
        tol: Incomplete = ...,
        hermitian: Incomplete = ...,
        **kwargs: Incomplete,
    ) -> None: ...
    def slogdet(self, dims: Incomplete = ..., **kwargs: Incomplete) -> None: ...
    def trace(
        self,
        dims: Incomplete = ...,
        *,
        offset: Incomplete = ...,
        dtype: Incomplete = ...,
        out: Incomplete = ...,
        **kwargs: Incomplete,
    ) -> None: ...
    def diagonal(
        self, dims: Incomplete = ..., *, offset: Incomplete = ..., **kwargs: Incomplete
    ) -> None: ...
    def solve(
        self, db: Incomplete, dims: Incomplete = ..., **kwargs: Incomplete
    ) -> None: ...
    def inv(self, dims: Incomplete = ..., **kwargs: Incomplete) -> None: ...

class EinopsAccessor:
    def __init__(self, xarray_obj: Incomplete) -> None: ...
    def rearrange(
        self, pattern: Incomplete, pattern_in: Incomplete = ..., **kwargs: Incomplete
    ) -> None: ...
    def reduce(
        self,
        pattern: Incomplete,
        reduction: Incomplete,
        pattern_in: Incomplete = ...,
        **kwargs: Incomplete,
    ) -> None: ...
