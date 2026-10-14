"""Accessors for xarray_einstats features."""

# pylint: disable=redefined-builtin

from collections.abc import Callable, Hashable
from typing import Any, Literal, final, overload

import xarray as xr

from ._types import Dims
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
    matmul,
    matrix_power,
    matrix_rank,
    matrix_transpose,
    norm,
    pinv,
    qr,
    slogdet,
    solve,
    svd,
    trace,
)


@xr.register_dataarray_accessor("linalg")
@final
class LinAlgAccessor:
    """Class that registers accessors for linalg functions to the DataArray class."""

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj

    def matrix_transpose(self, dims: Dims = None) -> xr.DataArray:
        """Call :func:`xarray_einstats.linalg.matrix_transpose` on this DataArray."""
        return matrix_transpose(self._obj, dims=dims)

    def matrix_power(self, n: int, dims: Dims = None, **kwargs: Any) -> xr.DataArray:
        """Call :func:`xarray_einstats.linalg.matrix_power` on this DataArray."""
        return matrix_power(self._obj, n, dims=dims, **kwargs)

    def matmul(self, other: xr.DataArray, dims: Dims = None, **kwargs: Any) -> xr.DataArray:
        """Call :func:`xarray_einstats.linalg.matmul` with this DataArray as ``a/da``."""
        return matmul(self._obj, other, dims=dims, **kwargs)

    def cholesky(self, dims: Dims = None, **kwargs: Any) -> xr.DataArray:
        """Call :func:`xarray_einstats.linalg.cholesky` on this DataArray."""
        return cholesky(self._obj, dims=dims, **kwargs)

    @overload
    def qr(
        self,
        dims: Dims = None,
        *,
        mode: Literal["reduced", "complete", "raw"] = "reduced",
        out_append: str = "2",
        **kwargs: Any,
    ) -> tuple[xr.DataArray, xr.DataArray]: ...

    @overload
    def qr(
        self, dims: Dims = None, *, mode: Literal["r"], out_append: str = "2", **kwargs: Any
    ) -> xr.DataArray: ...

    @overload
    def qr(
        self, dims: Dims = None, *, mode: str = "reduced", out_append: str = "2", **kwargs: Any
    ) -> tuple[xr.DataArray, xr.DataArray] | xr.DataArray: ...

    def qr(
        self, dims: Dims = None, *, mode: str = "reduced", out_append: str = "2", **kwargs: Any
    ) -> tuple[xr.DataArray, xr.DataArray] | xr.DataArray:
        """Call :func:`xarray_einstats.linalg.qr` on this DataArray."""
        return qr(self._obj, dims=dims, mode=mode, out_append=out_append, **kwargs)

    @overload
    def svd(
        self,
        dims: Dims = None,
        *,
        full_matrices: bool = True,
        compute_uv: Literal[True] = True,
        hermitian: bool = False,
        out_append: str = "2",
        **kwargs: Any,
    ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]: ...

    @overload
    def svd(
        self,
        dims: Dims = None,
        *,
        full_matrices: bool = True,
        compute_uv: Literal[False],
        hermitian: bool = False,
        out_append: str = "2",
        **kwargs: Any,
    ) -> xr.DataArray: ...

    @overload
    def svd(
        self,
        dims: Dims = None,
        *,
        full_matrices: bool = True,
        compute_uv: bool = True,
        hermitian: bool = False,
        out_append: str = "2",
        **kwargs: Any,
    ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray] | xr.DataArray: ...

    def svd(
        self,
        dims: Dims = None,
        *,
        full_matrices: bool = True,
        compute_uv: bool = True,
        hermitian: bool = False,
        out_append: str = "2",
        **kwargs: Any,
    ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray] | xr.DataArray:
        """Call :func:`xarray_einstats.linalg.svd` on this DataArray."""
        return svd(
            self._obj,
            dims=dims,
            full_matrices=full_matrices,
            compute_uv=compute_uv,
            hermitian=hermitian,
            out_append=out_append,
            **kwargs,
        )

    def eig(self, dims: Dims = None, **kwargs: Any) -> tuple[xr.DataArray, xr.DataArray]:
        """Call :func:`xarray_einstats.linalg.eig` on this DataArray."""
        return eig(self._obj, dims=dims, **kwargs)

    def eigh(
        self, dims: Dims = None, *, UPLO: Literal["L", "U"] = "L", **kwargs: Any
    ) -> tuple[xr.DataArray, xr.DataArray]:  # pylint: disable=invalid-name
        """Call :func:`xarray_einstats.linalg.eigh` on this DataArray."""
        return eigh(self._obj, dims=dims, UPLO=UPLO, **kwargs)

    def eigvals(self, dims: Dims = None, **kwargs: Any) -> xr.DataArray:
        """Call :func:`xarray_einstats.linalg.eigvals` on this DataArray."""
        return eigvals(self._obj, dims=dims, **kwargs)

    def eigvalsh(
        self, dims: Dims = None, *, UPLO: Literal["L", "U"] = "L", **kwargs: Any
    ) -> xr.DataArray:  # pylint: disable=invalid-name
        """Call :func:`xarray_einstats.linalg.eigvalsh` on this DataArray."""
        return eigvalsh(self._obj, dims=dims, UPLO=UPLO, **kwargs)

    def norm(
        self, dims: Dims = None, *, ord: Any = None, **kwargs: Any
    ) -> xr.DataArray:  # pylint: disable=redefined-builtin
        """Call :func:`xarray_einstats.linalg.norm` on this DataArray."""
        return norm(self._obj, dims=dims, ord=ord, **kwargs)

    def cond(self, dims: Dims = None, *, p: Any = None, **kwargs: Any) -> xr.DataArray:
        """Call :func:`xarray_einstats.linalg.cond` on this DataArray."""
        return cond(self._obj, dims=dims, p=p, **kwargs)

    def det(self, dims: Dims = None, **kwargs: Any) -> xr.DataArray:
        """Call :func:`xarray_einstats.linalg.det` on this DataArray."""
        return det(self._obj, dims=dims, **kwargs)

    def matrix_rank(
        self,
        dims: Dims = None,
        *,
        tol: float | xr.DataArray | None = None,
        rtol: float | xr.DataArray | None = None,
        hermitian: bool = False,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Call :func:`xarray_einstats.linalg.matrix_rank` on this DataArray."""
        return matrix_rank(self._obj, dims=dims, tol=tol, rtol=rtol, hermitian=hermitian, **kwargs)

    def slogdet(self, dims: Dims = None, **kwargs: Any) -> tuple[xr.DataArray, xr.DataArray]:
        """Call :func:`xarray_einstats.linalg.slogdet` on this DataArray."""
        return slogdet(self._obj, dims=dims, **kwargs)

    def trace(
        self,
        dims: Dims = None,
        *,
        offset: int = 0,
        dtype: Any = None,
        out: Any = None,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Call :func:`xarray_einstats.linalg.trace` on this DataArray."""
        return trace(self._obj, dims=dims, offset=offset, dtype=dtype, out=out, **kwargs)

    def diagonal(self, dims: Dims = None, *, offset: int = 0, **kwargs: Any) -> xr.DataArray:
        """Call :func:`xarray_einstats.linalg.diagonal` on this DataArray."""
        return diagonal(self._obj, dims=dims, offset=offset, **kwargs)

    def solve(self, db: xr.DataArray, dims: Dims = None, **kwargs: Any) -> xr.DataArray:
        """Call :func:`xarray_einstats.linalg.solve` with this DataArray as ``a/da``."""
        return solve(self._obj, db, dims=dims, **kwargs)

    def inv(self, dims: Dims = None, **kwargs: Any) -> xr.DataArray:
        """Call :func:`xarray_einstats.linalg.inv` on this DataArray."""
        return inv(self._obj, dims=dims, **kwargs)

    def pinv(
        self,
        dims: Dims = None,
        *,
        rcond: float | xr.DataArray | None = None,
        rtol: float | xr.DataArray | None = None,
        hermitian: bool = False,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Call :func:`xarray_einstats.linalg.pinv` on this DataArray."""
        return pinv(self._obj, dims=dims, rcond=rcond, rtol=rtol, hermitian=hermitian, **kwargs)


@xr.register_dataarray_accessor("einops")
@final
class EinopsAccessor:
    """Class that registers accessors for einops functions to the DataArray class."""

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj
        try:
            from .einops import rearrange, reduce

            self._rearrange = rearrange
            self._reduce = reduce
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                "`einops` library must be installed in order to use the einops accessor"
            ) from err

    def rearrange(
        self,
        pattern: str | list[Hashable | list[Any] | dict[Any, Any]],
        pattern_in: list[Hashable | dict[Any, Any]] | None = None,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Call :func:`xarray_einstats.einops.rearrange` on this DataArray."""
        return self._rearrange(self._obj, pattern=pattern, pattern_in=pattern_in, **kwargs)

    def reduce(
        self,
        pattern: str | list[str | list[Any] | dict[Any, Any]],
        reduction: str | Callable[..., Any],
        pattern_in: list[str | dict[Any, Any]] | None = None,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Call :func:`xarray_einstats.einops.reduce` on this DataArray."""
        return self._reduce(
            self._obj, pattern=pattern, reduction=reduction, pattern_in=pattern_in, **kwargs
        )
