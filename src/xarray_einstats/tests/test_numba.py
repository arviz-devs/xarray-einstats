# pylint: disable=redefined-outer-name, no-self-use
"""Test the numba module."""
import numpy as np
import pytest

from xarray_einstats.numba import histogram
from xarray_einstats.tutorial import generate_mcmc_like_dataset

from .utils import assert_dims_in_da, assert_dims_not_in_da


@pytest.fixture(scope="module")
def data():
    ds = generate_mcmc_like_dataset(5)
    return ds


class TestHistogram:
    @pytest.mark.parametrize("dims", ("team", ["team"], ("chain", "team")))
    def test_histogram(self, data, dims):
        out = histogram(data["mu"], dims)
        if isinstance(dims, str):
            dims = [dims]
        assert_dims_not_in_da(out, dims)
        assert_dims_in_da(out, ("draw", "bin"))
        assert "left_edges" in out.coords
        assert "right_edges" in out.coords
        assert np.allclose(out.left_edges.values[1:], out.right_edges.values[:-1])

    @pytest.mark.parametrize("bins", (10, "auto", np.arange(11)))
    def test_histogram_bins(self, data, bins):
        out = histogram(data["mu"], ("chain", "draw"), bins=bins)
        assert_dims_not_in_da(out, ("chain", "draw"))
        assert_dims_in_da(out, ("team", "bin"))
        if not isinstance(bins, str):
            assert len(out.bin) == 10
        assert "left_edges" in out.coords
        assert "right_edges" in out.coords
        if isinstance(bins, np.ndarray):
            assert np.allclose(out.left_edges, bins[:-1])
            assert np.allclose(out.right_edges, bins[1:])

    def test_histogram_density(self, data):
        out = histogram(data["mu"], ("chain", "draw"), bins=np.arange(10), density=True)
        assert_dims_not_in_da(out, ("chain", "draw"))
        assert_dims_in_da(out, ("team", "bin"))
        assert np.allclose(out.sum("bin"), 1)
