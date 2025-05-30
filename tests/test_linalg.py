# pylint: disable=redefined-outer-name, no-self-use, too-many-public-methods
"""Test the linalg module."""
import numpy as np
import packaging
import pytest
import xarray as xr
from xarray.testing import assert_allclose, assert_equal

from xarray_einstats import einsum, einsum_path, linalg, matmul, tutorial
from xarray_einstats.linalg import (
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
    pinv,
    qr,
    slogdet,
    solve,
    svd,
    trace,
)

from .utils import assert_dims_in_da, assert_dims_not_in_da


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


@pytest.fixture(scope="module")
def hermitian():
    """Hermitian and positive definite matrices."""
    # fmt: off
    a = [[[  2,  1],
          [  1,  2]],
         [[  5,  3],
          [  3,  5]],
         [[  4,-.7],
          [-.7,  4]]]
    # fmt: on
    da = xr.DataArray(a, dims=["batch", "dim", "dim2"])
    assert np.all(det(da, dims=("dim", "dim2")) > 0)
    return da


def test_no_dims(matrices, monkeypatch):
    with pytest.raises(TypeError, match="missing required argument dims"):
        inv(matrices)

    def default_dims(dims1, dims2):  # pylint: disable=unused-argument
        return ("dim", "dim2")

    monkeypatch.setattr(linalg, "get_default_dims", default_dims)

    out = inv(matrices)
    assert out.dims == matrices.dims


class TestEinsumFamily:
    # raw_einsum calls einsum, so the tests on raw_einsum also cover einsum, then
    # there are some specific ones for various reasons,
    # mostly to test features supported only in einsum
    def test_raw_einsum_implicit(self, matrices):
        out = einsum("batch,experiment", matrices, matrices)
        da = matrices.sum("experiment") * matrices.sum("batch")
        assert list(out.dims) == ["dim", "dim2", "experiment", "batch"]
        assert_allclose(out, da.transpose(*out.dims))

    def test_raw_einsum_explicit(self, matrices):
        out = einsum("batch,experiment->", matrices, matrices)
        da = (matrices.sum("experiment") * matrices.sum("batch")).sum(("batch", "experiment"))
        assert_dims_not_in_da(out, ["batch", "experiment"])
        assert_dims_in_da(out, ["dim", "dim2"])
        assert_allclose(out, da.transpose(*out.dims))

    def test_raw_einsum_transpose(self, matrices):
        out = einsum("batch experiment->experiment batch", matrices)
        assert out.ndim == matrices.ndim
        assert out.dims[-1] == "batch"
        assert out.dims[-2] == "experiment"
        assert_allclose(out, matrices.transpose(*out.dims))

    def test_einsum_outer(self, matrices):
        out = einsum([[], []], matrices, matrices, keep_dims={"dim"}, out_append="_bis{i}")
        da = (matrices * matrices.rename(dim="dim_bis2")).transpose(*out.dims)
        assert list(out.dims) == ["batch", "experiment", "dim2", "dim", "dim_bis2"]
        assert_allclose(out, da)

    def test_einsum_implicit(self, matrices):
        da = matrices.rename(batch="ba tch", experiment="exp,er->iment")
        out = einsum([["ba tch"], ["exp,er->iment"]], da, da)
        da = da.sum("exp,er->iment") * da.sum("ba tch")
        assert_dims_in_da(out, ["dim", "dim2", "ba tch", "exp,er->iment"])
        assert_allclose(out, da.transpose(*out.dims))

    def test_einsum_path(self, matrices):
        out = einsum_path([["batch"], ["experiment"], []], matrices, matrices)
        assert out


class TestWrappers:
    @pytest.mark.parametrize(
        "method",
        (norm, cond, det, matrix_rank, trace),
    )
    def test_reduce_to_scalar(self, matrices, method):
        out = method(matrices, dims=("dim", "dim2"))
        assert out.shape == (10, 3)
        assert_dims_in_da(out, ("batch", "experiment"))
        assert_dims_not_in_da(out, ["dim", "dim2"])

    @pytest.mark.parametrize("kind", ("tol", "rtol"))
    def test_matrix_rank_dataarray_tol(self, matrices, kind):
        kwargs = {kind: xr.DataArray([1e-2, 1e-3, 1e-5], dims=["experiment"])}
        out = matrix_rank(matrices, dims=("dim", "dim2"), **kwargs)
        assert out.shape == (10, 3)
        assert_dims_in_da(out, ("batch", "experiment"))
        assert_dims_not_in_da(out, ["dim", "dim2"])

    def test_vector_norm(self, matrices):
        out = norm(matrices, dims="experiment")
        assert_dims_in_da(out, ("batch", "dim", "dim2"))
        assert_dims_not_in_da(out, ["experiment"])

    def test_inv(self, matrices):
        out = inv(matrices, dims=("dim", "dim2"))
        assert out.shape == matrices.shape
        assert out.dims == matrices.dims

    def test_pinv(self, matrices):
        out = pinv(matrices, dims=("experiment", "dim"))
        out_dims_exp = ("batch", "dim2", "dim", "experiment")
        assert out.dims == out_dims_exp
        assert out.shape == tuple(out.sizes[dim] for dim in out_dims_exp)

    @pytest.mark.parametrize("kind", ("rcond", "rtol"))
    def test_pinv_dataarray_tol(self, matrices, kind):
        kwargs = {kind: xr.DataArray([1e-2, 1e-3, 1e-5], dims=["experiment"])}
        out = pinv(matrices, dims=("batch", "dim"), **kwargs)
        out_dims_exp = ("experiment", "dim2", "dim", "batch")
        assert out.dims == out_dims_exp
        assert out.shape == tuple(out.sizes[dim] for dim in out_dims_exp)

    def test_transpose(self, hermitian):
        assert_equal(hermitian, matrix_transpose(hermitian, dims=("dim", "dim2")))

    def test_matrix_power(self, matrices):
        out = matrix_power(matrices, 2, dims=("dim", "dim2"))
        assert out.shape == matrices.shape
        assert out.dims == matrices.dims

    def test_matmul_dims2(self, matrices):
        out = matmul(matrices, matrices, dims=("dim", "dim2"))
        assert out.shape == matrices.shape
        assert out.dims == matrices.dims

    def test_matmul_dims3(self):
        rng = np.random.default_rng(3)
        da = xr.DataArray(rng.normal(size=(2, 3, 5, 7)), dims=["m", "n", "l", "p"])
        db = da.rename(m="m_bis")
        out = matmul(da, db, dims=("m", "n", "m_bis"))
        assert out.shape == (5, 7, 2, 2)
        assert_dims_not_in_da(out, ["n"])
        assert_dims_in_da(out, ("m", "m_bis", "l", "p"))

    def test_matmul_dims3_rename(self):
        rng = np.random.default_rng(3)
        da = xr.DataArray(rng.normal(size=(2, 3, 5, 7)), dims=["m", "n", "l", "p"])
        out = matmul(da, da, dims=("m", "n", "m"))
        assert out.shape == (5, 7, 2, 2)
        assert_dims_not_in_da(out, ["n"])
        assert_dims_in_da(out, ("m", "m2", "l", "p"))

    def test_matmul_dims22(self):
        rng = np.random.default_rng(3)
        da = xr.DataArray(rng.normal(size=(2, 3, 5, 7)), dims=["m", "n", "l", "p"])
        db = da.rename(n="n_bis", m="n")
        out = matmul(da, db, dims=(("m", "n"), ("n_bis", "n")))
        assert out.shape == (5, 7, 2, 2)
        assert_dims_not_in_da(out, ["n_bis"])
        assert_dims_in_da(out, ["m", "n", "l", "p"])

    def test_matmul_dims22_rename(self):
        rng = np.random.default_rng(3)
        da = xr.DataArray(rng.normal(size=(2, 3, 5, 7)), dims=["m", "n", "l", "p"])
        db = da.rename(n="n_bis")
        out = matmul(da, db, dims=(("m", "n"), ("n_bis", "m")))
        assert out.shape == (5, 7, 2, 2)
        assert_dims_not_in_da(out, ["n_bis", "n"])
        assert_dims_in_da(out, ["m", "m2", "l", "p"])

    def test_inv_matmul(self, matrices):
        aux = inv(matrices, dims=("dim", "dim2"))
        out = matmul(matrices, aux, dims=("dim", "dim2"))
        assert out.shape == matrices.shape
        assert out.dims == matrices.dims
        np.testing.assert_allclose(
            np.eye(len(out.dim)), out.isel(experiment=0, batch=0).values, atol=1e-14, rtol=1e-7
        )

    def test_eig_funcs(self, matrices):
        eig_w, eig_v = eig(matrices, dims=("dim", "dim2"))
        eigvals_w = eigvals(matrices, dims=("dim", "dim2"))
        assert_allclose(eig_w, eigvals_w, atol=1e-15)
        left = matmul(matrices, eig_v, dims=("dim", "dim2"))
        right = eig_w * eig_v
        assert_allclose(left, right.transpose(*left.dims), atol=1e-14)

    def test_eigh_funcs(self, hermitian):
        eig_w, eig_v = eigh(hermitian, dims=("dim", "dim2"))
        eigvals_w = eigvalsh(hermitian, dims=("dim", "dim2"))
        assert_allclose(eig_w, eigvals_w, atol=1e-15)
        left = matmul(hermitian, eig_v, dims=("dim", "dim2"))
        right = eig_w * eig_v
        assert_allclose(left, right.transpose(*left.dims), atol=1e-14)

    def test_cholesky(self, hermitian):
        chol = cholesky(hermitian, dims=("dim", "dim2"))
        assert hermitian.dims == chol.dims
        chol_chol_t = matmul(
            chol, matrix_transpose(chol, dims=("dim", "dim2")), dims=("dim", "dim2")
        )
        assert_allclose(hermitian, chol_chol_t)

    @pytest.mark.skipif(
        packaging.version.Version(np.__version__) < packaging.version.Version("1.22"),
        reason="Requires numpy>=1.22 to support batched qr",
    )
    def test_qr(self, matrices):
        q_da, r_da = qr(matrices, dims=("dim", "dim2"))
        assert_allclose(matrices, matmul(q_da, r_da, dims=("dim", "dim2")))

    @pytest.mark.skipif(
        packaging.version.Version(np.__version__) < packaging.version.Version("1.22"),
        reason="Requires numpy>=1.22 to support batched qr",
    )
    def test_qr_non_square(self, matrices):
        q_da, r_da = qr(matrices, dims=("experiment", "dim2"))
        assert_allclose(
            matrices,
            matmul(
                q_da, r_da, dims=[["experiment", "experiment2"], ["experiment", "dim2"]]
            ).transpose(*matrices.dims),
        )

    def test_svd(self, matrices):
        u_da, s_da, vh_da = svd(matrices, dims=("dim", "dim2"), out_append="_bis")
        s_full = xr.zeros_like(matrices)
        idx = xr.DataArray(np.arange(len(matrices["dim"])), dims="pointwise_sel")
        s_full.loc[{"dim": idx, "dim2": idx}] = s_da.rename(dim="pointwise_sel")
        compare = matmul(
            matmul(u_da, s_full, dims=[["dim", "dim_bis"], ["dim", "dim2"]]),
            vh_da,
            dims=("dim", "dim2", "dim2_bis"),
        ).rename(dim2_bis="dim2")
        assert_allclose(matrices, compare.transpose(*matrices.dims), atol=1e-13)

    def test_svd_non_square(self, matrices):
        u_da, s_da, vh_da = svd(matrices, dims=("experiment", "dim"), out_append="_bis")
        s_full = xr.zeros_like(matrices)
        # experiment is shorter than dim
        idx = xr.DataArray(np.arange(len(matrices["experiment"])), dims="pointwise_sel")
        s_full.loc[{"experiment": idx, "dim": idx}] = s_da.rename(experiment="pointwise_sel")
        compare = matmul(
            matmul(u_da, s_full, dims=[["experiment", "experiment_bis"], ["experiment", "dim"]]),
            vh_da,
            dims=("experiment", "dim", "dim_bis"),
        ).rename(dim_bis="dim")
        assert_allclose(matrices, compare.transpose(*matrices.dims), atol=1e-13)

    def test_slogdet_det(self, matrices):
        sign, logdet = slogdet(matrices, dims=("dim", "dim2"))
        det_da = det(matrices, dims=("dim", "dim2"))
        assert_allclose(sign * np.exp(logdet), det_da)

    def test_solve_two_dims(self, matrices):
        b = matrices.std("dim2")
        y = solve(matrices, b, dims=("dim", "dim2"))
        assert_allclose(b, xr.dot(matrices, y, dim="dim2"), atol=1e-14)

    def test_solve_three_dims(self, matrices):
        b = matrices.std("dim2")
        a = matrices.isel(batch=0)
        y = solve(a, b, dims=("dim", "dim2", "batch"))
        assert_allclose(b, xr.dot(a, y, dim="dim2").transpose(*b.dims), atol=1e-14)

    def test_diagonal(self, matrices):
        idx = xr.DataArray(np.arange(len(matrices["dim"])), dims="pointwise_sel")
        diag = diagonal(matrices, dims=("dim", "dim2"))
        diag_compare = (
            matrices.isel(dim=idx, dim2=idx).rename(pointwise_sel="dim").transpose(*diag.dims)
        )
        assert_allclose(diag, diag_compare)
