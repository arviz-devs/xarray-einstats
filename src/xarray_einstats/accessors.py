"""Accessors for xarray_einstats features."""

import xarray as xr

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


@xr.register_dataarray_accessor("linalg")
class LinAlgAccessor:
    """Class that registers accessors for linalg functions to the DataArray class."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def matrix_transpose(self, dims):
        """Call :func:`xarray_einstats.linalg.matrix_transpose` on this DataArray."""
        return matrix_transpose(self._obj, dims=dims)

    def matrix_power(self, n, dims=None, **kwargs):
        """Call :func:`xarray_einstats.linalg.matrix_power` on this DataArray."""
        return matrix_power(self._obj, n, dims=dims, **kwargs)

    def cholesky(self, dims=None, **kwargs):
        """Call :func:`xarray_einstats.linalg.cholesky` on this DataArray."""
        return cholesky(self._obj, dims=dims, **kwargs)

    def qr(self, dims=None, *, mode="reduced", out_append="2", **kwargs):
        """Call :func:`xarray_einstats.linalg.qr` on this DataArray."""
        return qr(self._obj, dims=dims, mode=mode, out_append=out_append, **kwargs)

    def svd(
        self,
        dims=None,
        *,
        full_matrices=True,
        compute_uv=True,
        hermitian=False,
        out_append="2",
        **kwargs,
    ):
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

    def eig(self, dims=None, **kwargs):
        """Call :func:`xarray_einstats.linalg.eig` on this DataArray."""
        return eig(self._obj, dims=dims, **kwargs)

    def eigh(self, dims=None, *, UPLO="L", **kwargs):  # pylint: disable=invalid-name
        """Call :func:`xarray_einstats.linalg.eigh` on this DataArray."""
        return eigh(self._obj, dims=dims, UPLO=UPLO, **kwargs)

    def eigvals(self, dims=None, **kwargs):
        """Call :func:`xarray_einstats.linalg.eigvals` on this DataArray."""
        return eigvals(self._obj, dims=dims, **kwargs)

    def eigvalsh(self, dims=None, *, UPLO="L", **kwargs):  # pylint: disable=invalid-name
        """Call :func:`xarray_einstats.linalg.eigvalsh` on this DataArray."""
        return eigvalsh(self._obj, dims=dims, UPLO=UPLO, **kwargs)

    def norm(self, dims=None, *, ord=None, **kwargs):  # pylint: disable=redefined-builtin
        """Call :func:`xarray_einstats.linalg.norm` on this DataArray."""
        return norm(self._obj, dims=dims, ord=ord, **kwargs)

    def cond(self, dims=None, *, p=None, **kwargs):
        """Call :func:`xarray_einstats.linalg.cond` on this DataArray."""
        return cond(self._obj, dims=dims, p=p, **kwargs)

    def det(self, dims=None, **kwargs):
        """Call :func:`xarray_einstats.linalg.det` on this DataArray."""
        return det(self._obj, dims=dims, **kwargs)

    def matrix_rank(self, dims=None, *, tol=None, hermitian=False, **kwargs):
        """Call :func:`xarray_einstats.linalg.matrix_rank` on this DataArray."""
        return matrix_rank(self._obj, dims=dims, tol=tol, hermitian=hermitian, **kwargs)

    def slogdet(self, dims=None, **kwargs):
        """Call :func:`xarray_einstats.linalg.slogdet` on this DataArray."""
        return slogdet(self._obj, dims=dims, **kwargs)

    def trace(self, dims=None, *, offset=0, dtype=None, out=None, **kwargs):
        """Call :func:`xarray_einstats.linalg.trace` on this DataArray."""
        return trace(self._obj, dims=dims, offset=offset, dtype=dtype, out=out, **kwargs)

    def diagonal(self, dims=None, *, offset=0, **kwargs):
        """Call :func:`xarray_einstats.linalg.diagonal` on this DataArray."""
        return diagonal(self._obj, dims=dims, offset=offset, **kwargs)

    def solve(self, db, dims=None, **kwargs):
        """Call :func:`xarray_einstats.linalg.solve` with this DataArray as ``a/da``."""
        return solve(self._obj, db, dims=dims, **kwargs)

    def inv(self, dims=None, **kwargs):
        """Call :func:`xarray_einstats.linalg.inv` on this DataArray."""
        return inv(self._obj, dims=dims, **kwargs)


@xr.register_dataarray_accessor("einops")
class EinopsAccessor:
    """Class that registers accessors for einops functions to the DataArray class."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        try:
            from .einops import rearrange, reduce

            self._rearrange = rearrange
            self._reduce = reduce
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                "`einops` library must be installed in order to use the einops accessor"
            ) from err

    def rearrange(self, pattern, pattern_in=None, **kwargs):
        """Call :func:`xarray_einstats.einops.rearrange` on this DataArray."""
        return self._rearrange(self._obj, pattern=pattern, pattern_in=pattern_in, **kwargs)

    def reduce(self, pattern, reduction, pattern_in=None, **kwargs):
        """Call :func:`xarray_einstats.einops.reduce` on this DataArray."""
        return self._reduce(
            self._obj, pattern=pattern, reduction=reduction, pattern_in=pattern_in, **kwargs
        )
