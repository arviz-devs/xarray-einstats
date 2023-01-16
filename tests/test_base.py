# pylint: disable=redefined-outer-name, no-self-use
"""Test top level functions."""
import numpy as np
import pytest

from xarray_einstats import empty_ref, ones_ref, sort, zeros_ref
from xarray_einstats.tutorial import generate_matrices_dataarray, generate_mcmc_like_dataset

from .utils import assert_dims_in_da, assert_dims_not_in_da


@pytest.fixture(scope="module")
def dataset():
    return generate_mcmc_like_dataset(324)


def test_sort(dataset):
    res = sort(dataset["mu"], dim="team")
    assert_dims_in_da(res, ["chain", "draw", "team"])
    team_diff = res.isel(team=slice(1, None)).values - res.isel(team=slice(None, -1)).values
    assert np.all(team_diff > 0)
    chain_diff = res.isel(chain=slice(1, None)).values - res.isel(chain=slice(None, -1)).values
    assert not np.all(chain_diff > 0)


@pytest.mark.parametrize("fun", (zeros_ref, ones_ref, empty_ref))
class TestCreators:
    def test_da_input(self, fun, dataset):
        res = fun(dataset["mu"], dims=["draw", "team"])
        assert res.ndim == 2
        assert res.dtype == dataset["mu"].dtype
        assert_dims_in_da(res, ["draw", "team"])
        assert_dims_not_in_da(res, ["chain", "match"])

    def test_ds_input(self, fun, dataset):
        res = fun(dataset, dims=["draw", "team", "match"])
        assert res.ndim == 3
        assert_dims_in_da(res, ["draw", "team", "match"])
        assert_dims_not_in_da(res, ["chain"])

    def test_multiple_inputs(self, fun, dataset):
        da = generate_matrices_dataarray(43)
        res = fun(dataset, da, dims=["draw", "team", "batch"])
        assert res.ndim == 3
        assert_dims_in_da(res, ["draw", "team", "batch"])
        assert_dims_not_in_da(res, ["chain", "match", "subject"])
