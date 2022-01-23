# pylint: disable=redefined-outer-name, no-self-use
"""Test the stats module."""
import numpy as np
import pytest
from scipy import stats

from xarray_einstats import tutorial
from xarray_einstats.stats import XrContinuousRV, XrDiscreteRV, rankdata


@pytest.fixture(scope="module")
def data():
    return tutorial.generate_mcmc_like_dataset(3)


@pytest.mark.parametrize("wrapper", ("continuous", "discrete"))
class TestRvWrappers:
    @pytest.mark.parametrize(
        "method", ("pxf", "logpxf", "cdf", "logcdf", "sf", "logsf", "ppf", "isf")
    )
    def test_xk_methods_array(self, data, wrapper, method):
        if wrapper == "continuous":
            dist = XrContinuousRV(stats.norm, data["mu"], data["sigma"])
            if "pxf" in method:
                method = method.replace("x", "d")
        else:
            dist = XrDiscreteRV(stats.poisson, data["mu"], data["sigma"])
            if "pxf" in method:
                method = method.replace("x", "m")
        vals = np.linspace(0, 1, 10)
        meth = getattr(dist, method)
        out = meth(vals)
        assert out.ndim == 4
        if method in {"ppf", "isf"}:
            assert "quantile" in out.dims
        else:
            assert "point" in out.dims

    @pytest.mark.parametrize(
        "method", ("pxf", "logpxf", "cdf", "logcdf", "sf", "logsf", "ppf", "isf")
    )
    def test_xk_methods_dataarray(self, data, wrapper, method):
        if wrapper == "continuous":
            dist = XrContinuousRV(stats.norm, data["mu"], data["sigma"])
            if "pxf" in method:
                method = method.replace("x", "d")
        else:
            dist = XrDiscreteRV(stats.poisson, data["mu"], data["sigma"])
            if "pxf" in method:
                method = method.replace("x", "m")
        meth = getattr(dist, method)
        out = meth(data["x_plot"])
        assert out.ndim == 4
        assert "plot_dim" in out.dims


@pytest.mark.parametrize("dims", ("match", ("chain", "draw"), None))
def test_rankdata(data, dims):
    da = data["score"]
    out = rankdata(da, dims=dims)
    if isinstance(dims, str):
        rank_size = len(da[dims])
    elif dims is None:
        rank_size = da.size
    else:
        rank_size = np.prod([len(da[dim]) for dim in dims])
    assert np.all(out <= rank_size)
