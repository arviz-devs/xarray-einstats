# pylint: disable=redefined-outer-name, no-self-use
"""Test the stats module."""
import numpy as np
import pytest
import xarray as xr
from scipy import stats
from xarray.testing import assert_allclose

from xarray_einstats import tutorial
from xarray_einstats.stats import (
    XrContinuousRV,
    XrDiscreteRV,
    circmean,
    circstd,
    circvar,
    gmean,
    hmean,
    kurtosis,
    median_abs_deviation,
    rankdata,
    skew,
)

from .utils import assert_dims_in_da, assert_dims_not_in_da


@pytest.fixture(scope="module")
def data():
    return tutorial.generate_mcmc_like_dataset(3)


@pytest.mark.parametrize("wrapper", ("continuous", "discrete"))
class TestRvWrappers:
    @pytest.mark.parametrize(
        "method", ("pxf", "logpxf", "cdf", "logcdf", "sf", "logsf", "ppf", "isf")
    )
    def test_eval_methods_array(self, data, wrapper, method):
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
    def test_eval_methods_dataarray(self, data, wrapper, method):
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

    @pytest.mark.parametrize("dim_names", (None, ("name1", "name2")))
    def test_rv_method(self, data, wrapper, dim_names):
        if wrapper == "continuous":
            dist = XrContinuousRV(stats.norm, data["mu"], data["sigma"])
        else:
            dist = XrDiscreteRV(stats.poisson, data["mu"], data["sigma"])
        out = dist.rvs(size=(2, 7), dims=dim_names)
        if dim_names is None:
            dim_names = ["rv_dim0", "rv_dim1"]
        assert_dims_in_da(out, dim_names)
        assert len(out[dim_names[0]]) == 2
        assert len(out[dim_names[1]]) == 7

    @pytest.mark.parametrize("size", (1, 10))
    @pytest.mark.parametrize("dims", (None, "name", ["name"]))
    def test_rv_method_scalar_size(self, data, wrapper, size, dims):
        if wrapper == "continuous":
            dist = XrContinuousRV(stats.norm, data["mu"], data["sigma"])
        else:
            dist = XrDiscreteRV(stats.poisson, data["mu"], data["sigma"])
        out = dist.rvs(size=size, dims=dims)
        dim_name = "rv_dim0" if dims is None else "name"
        if size == 1:
            assert dim_name not in out.dims
        else:
            assert dim_name in out.dims
            assert len(out[dim_name]) == size

    def test_non_broadcastable_input(self, data, wrapper):
        if wrapper == "continuous":
            dist = XrContinuousRV(stats.norm, data["mu"], 1)
        else:
            dist = XrDiscreteRV(stats.poisson, data["mu"], 1)
        out = dist.cdf([1, 2])
        expected_dims = [*data["mu"].dims, "point"]
        assert out.ndim == len(expected_dims)
        assert_dims_in_da(out, expected_dims)

    def test_kwargs_input(self, data, wrapper):
        if wrapper == "continuous":
            dist1 = XrContinuousRV(stats.norm, data["mu"], 1)
            dist2 = XrContinuousRV(stats.norm, loc=data["mu"], scale=1)
        else:
            dist1 = XrDiscreteRV(stats.poisson, data["mu"], 1)
            dist2 = XrDiscreteRV(stats.poisson, mu=data["mu"], loc=1)
        out1 = dist1.cdf([1, 2])
        out2 = dist2.cdf([1, 2])
        expected_dims = [*data["mu"].dims, "point"]
        assert out2.ndim == len(expected_dims)
        assert_dims_in_da(out2, expected_dims)
        assert_allclose(out1, out2, atol=1e-15)


@pytest.mark.parametrize("xr_obj", ("DataArray", "Dataset"))
class TestStats:
    @pytest.mark.parametrize("dims", ("chain", ("chain", "draw"), None))
    def test_rankdata(self, data, dims, xr_obj):
        if xr_obj == "DataArray":
            xr_in = data["score"]
        else:
            if dims is None:
                pytest.skip("rankdata doesn't support Dataset input and dims=None")
            xr_in = data[["mu", "sigma"]]
        xr_out = rankdata(xr_in, dims=dims)
        if xr_obj == "DataArray":
            assert isinstance(xr_out, xr.DataArray)
            xr_out = xr_out.to_dataset()
            xr_in = xr_in.to_dataset()
        for var_name in xr_in.data_vars:
            da = xr_in[var_name]
            out = xr_out[var_name]
            if isinstance(dims, str):
                rank_size = len(da[dims])
            elif dims is None:
                rank_size = da.size
            else:
                rank_size = np.prod([len(da[dim]) for dim in dims])
            assert np.all(out <= rank_size)

    @pytest.mark.parametrize("dims", ("chain", ("chain", "draw"), None))
    @pytest.mark.parametrize(
        "func", (gmean, hmean, circmean, circstd, circvar, kurtosis, skew, median_abs_deviation)
    )
    def test_reduce_function(self, data, dims, func, xr_obj):
        if xr_obj == "DataArray":
            xr_in = data["mu"]
        else:
            xr_in = data[["mu", "sigma"]]
        xr_out = func(xr_in, dims=dims)
        if xr_obj == "DataArray":
            assert isinstance(xr_out, xr.DataArray)
            xr_out = xr_out.to_dataset()
            xr_in = xr_in.to_dataset()
        for var_name in xr_in.data_vars:
            da = xr_in[var_name]
            out = xr_out[var_name]
            if dims is None:
                dims = da.dims
            elif isinstance(dims, str):
                dims = [dims]
            expected_dims = [dim for dim in da.dims if dim not in dims]
            assert_dims_in_da(out, expected_dims)
            assert_dims_not_in_da(out, dims)


def test_mad_da_scale(data):
    s_da = xr.DataArray([1, 2, 1, 1], coords={"chain": data.chain})
    out = median_abs_deviation(data["mu"], dims="draw", scale=s_da)
    out1 = median_abs_deviation(data["mu"].sel(chain=0), dims="draw", scale=1)
    out2 = median_abs_deviation(data["mu"].sel(chain=1), dims="draw", scale=2)
    assert_dims_in_da(out, ("chain", "team"))
    assert_dims_not_in_da(out, ["draw"])
    assert_allclose(out.sel(chain=0), out1)
    assert_allclose(out.sel(chain=1), out2)
