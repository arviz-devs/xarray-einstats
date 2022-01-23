# pylint: disable=redefined-outer-name, no-self-use
import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_allclose

from xarray_einstats import einsum, matmul, raw_einsum, tutorial
from xarray_einstats.linalg import (
    cholesky,
    cond,
    det,
    eig,
    eigh,
    eigvals,
    eigvalsh,
    inv,
    matrix_power,
    matrix_rank,
    norm,
    qr,
    slogdet,
    solve,
    svd,
    trace,
)


def assert_dims_in_da(da, dims):
    for dim in dims:
        assert dim in da.dims


def assert_dims_not_in_da(da, dims):
    for dim in dims:
        assert dim not in da.dims


@pytest.fixture(scope="module")
def data():
    ds = xr.Dataset(
        {
            "a": (("chain", "draw", "team"), np.full((4, 5, 3), 2)),
            "b": (("chain", "draw"), np.ones((4, 5))),
        }
    )
    return ds


@pytest.fixture(scope="module")
def matrices():
    ds = tutorial.generate_matrices_dataarray(3)
    return ds


class TestEinsumFamily:
    # raw_einsum calls einsum, so the tests on raw_einsum also cover einsum, then
    # there are some specific ones for various reasons,
    # mostly to test features supported only in einsum
    def test_raw_einsum_implicit(self, matrices):
        out = raw_einsum("batch,experiment", matrices, matrices)
        da = (matrices.sum("experiment") * matrices.sum("batch"))
        assert list(out.dims) == ["dim", "dim2", "experiment", "batch"]
        assert_allclose(out, da.transpose(*out.dims))

    def test_raw_einsum_explicit(self, matrices):
        out = raw_einsum("batch,experiment->", matrices, matrices)
        da = (matrices.sum("experiment") * matrices.sum("batch")).sum(("batch", "experiment"))
        assert_dims_not_in_da(out, ["batch", "experiment"])
        assert_dims_in_da(out, ["dim", "dim2"])
        assert_allclose(out, da.transpose(*out.dims))

    def test_einsum_outer(self, matrices):
        out = einsum([[], []], matrices, matrices, keep_dims={"dim"}, out_append="_bis{i}")
        da = (matrices * matrices.rename(dim="dim_bis2")).transpose(*out.dims)
        assert list(out.dims) == ["batch", "experiment", "dim2", "dim", "dim_bis2"]
        assert_allclose(out, da)

    def test_einsum_implicit(self, matrices):
        da = matrices.rename(batch="ba tch", experiment="exp,er->iment")
        out = einsum([["ba tch"], ["exp,er->iment"]], da, da)
        da = (da.sum("exp,er->iment") * da.sum("ba tch"))
        assert_dims_in_da(out, ["dim", "dim2", "ba tch", "exp,er->iment"])
        assert_allclose(out, da.transpose(*out.dims))


class TestWrappers:
    @pytest.mark.parametrize(
        "method",
        (norm, cond, det, matrix_rank, trace),
    )
    def test_reduce_to_scalar(self, matrices, method):
        out = method(matrices, dims=("dim", "dim2"))
        assert out.shape == (10, 3)
        assert "batch" in out.dims
        assert "experiment" in out.dims
        assert "dim" not in out.dims
        assert "dim2" not in out.dims

    def test_inv(self, matrices):
        out = inv(matrices, dims=("dim", "dim2"))
        assert out.shape == matrices.shape
        assert out.dims == matrices.dims

    def test_matrix_power(self, matrices):
        out = matrix_power(matrices, 2, dims=("dim", "dim2"))
        assert out.shape == matrices.shape
        assert out.dims == matrices.dims

    def test_matmul(self, matrices):
        out = matmul(matrices, matrices, dims=("dim", "dim2"))
        assert out.shape == matrices.shape
        assert out.dims == matrices.dims

    def test_inv_matmul(self, matrices):
        aux = inv(matrices, dims=("dim", "dim2"))
        out = matmul(matrices, aux, dims=("dim", "dim2"))
        assert out.shape == matrices.shape
        assert out.dims == matrices.dims
        np.testing.assert_allclose(
            np.eye(len(out.dim)), out.isel(experiment=0, batch=0).values, atol=1e-14, rtol=1e-7
        )
