# pylint: disable=redefined-outer-name, no-self-use
"""Test the accessors."""

from importlib.util import find_spec

import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_allclose

import xarray_einstats
from xarray_einstats import linalg
from xarray_einstats.accessors import (  # pylint: disable=unused-import
    EinopsAccessor,
    LinAlgAccessor,
)


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
    assert np.all(linalg.det(da, dims=("dim", "dim2")) > 0)
    return da


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(9)
    da = xr.DataArray(
        rng.normal(size=(4, 6, 15, 8)), dims=["batch", "subject", "experiment", "drug"]
    )
    return da


@pytest.mark.parametrize(
    "func",
    (
        "cholesky",
        "cond",
        "det",
        "diagonal",
        "eig",
        "eigh",
        "eigvals",
        "eigvalsh",
        "inv",
        "matrix_rank",
        "norm",
        "qr",
        "slogdet",
        "svd",
        "trace",
    ),
)
def test_linalg_accessor(hermitian, func):
    da_ac = getattr(hermitian.linalg, func)(dims=("dim", "dim2"))
    da_fun = getattr(linalg, func)(hermitian, dims=("dim", "dim2"))
    if isinstance(da_ac, tuple):
        for d_ac, d_fun in zip(da_ac, da_fun):
            assert_allclose(d_ac, d_fun)
    else:
        assert_allclose(da_ac, da_fun)


def test_linalg_accessor_solve(hermitian):
    db = hermitian.std("dim2")
    da_ac = hermitian.linalg.solve(db, dims=("dim", "dim2"))
    da_fun = linalg.solve(hermitian, db, dims=("dim", "dim2"))
    assert_allclose(da_ac, da_fun)


@pytest.mark.skipif(find_spec("einops") is None, reason="einops must be installed")
def test_einops_accessor_rearrange(data):
    pattern = "(e1 e2)=experiment -> e1 e2"
    kwargs = {"e1": 3, "e2": 5}
    da_fun = xarray_einstats.einops.rearrange(data, pattern, **kwargs)
    da_ac = data.einops.rearrange(pattern, **kwargs)
    assert_allclose(da_fun, da_ac)


@pytest.mark.skipif(find_spec("einops") is None, reason="einops must be installed")
def test_einops_accessor_reduce(data):
    pattern_in = [{"batch (hh.mm)": ["d1", "d2"]}]
    pattern = ["d1", "subject"]
    kwargs = {"d2": 2}
    input_data = data.rename({"batch": "batch (hh.mm)"})
    da_fun = xarray_einstats.einops.reduce(
        input_data,
        pattern,
        reduction="mean",
        pattern_in=pattern_in,
        **kwargs,
    )
    da_ac = input_data.einops.reduce(pattern, "mean", pattern_in=pattern_in, **kwargs)
    assert_allclose(da_fun, da_ac)
