# pylint: disable=redefined-outer-name, no-self-use, too-many-positional-arguments
"""Test the stats module."""
import warnings

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
    logsumexp,
    median_abs_deviation,
    multivariate_normal,
    rankdata,
    skew,
)

from .utils import assert_dims_in_da, assert_dims_not_in_da


@pytest.fixture(scope="module")
def data():
    return tutorial.generate_mcmc_like_dataset(3)


@pytest.fixture(scope="module")
def stacked_data(data):
    return data["mu"].rename(chain="aux").expand_dims(aux2=2).stack(chain=("aux", "aux2"))


def get_dist_and_clean_method(wrapper, data, method=None, in_type="array"):
    par2 = 1 if in_type == "scalar" else data["sigma"]
    if wrapper in ("continuous", "preliz"):
        if wrapper == "continuous":
            raw_dist = stats.norm
        else:
            try:
                from preliz import Normal

                raw_dist = Normal
            except ImportError:
                pytest.skip("PreliZ not installed")
        dist = XrContinuousRV(raw_dist, data["mu"], par2)
    elif wrapper == "rv_continuous":
        dist = XrContinuousRV(stats.Normal, mu=data["mu"], sigma=par2)
    elif wrapper == "discrete":
        dist = XrDiscreteRV(stats.poisson, data["mu"], par2)
    else:
        raise ValueError(f"Unkown value for 'wrapper': {wrapper}")
    if method is not None and "pxf" in method:
        method = method.replace("x", "m" if wrapper == "discrete" else "d")
    if method is not None:
        return dist, method
    return dist


@pytest.mark.parametrize("wrapper", ("continuous", "discrete", "preliz", "rv_continuous"))
class TestRvWrappers:
    @pytest.mark.parametrize(
        "method", ("pxf", "logpxf", "cdf", "logcdf", "sf", "logsf", "ppf", "isf")
    )
    def test_eval_methods_scalar(self, data, wrapper, method):
        dist, method = get_dist_and_clean_method(wrapper, data, method)
        meth = getattr(dist, method)
        with warnings.catch_warnings():
            if wrapper == "preliz" and method in ("logcdf", "logsf"):
                warnings.filterwarnings(
                    "ignore", message="divide by zero encountered in log", category=RuntimeWarning
                )
            out = meth(0.9)
        assert out.ndim == 3
        assert_dims_not_in_da(out, ["quantile", "point"])

    @pytest.mark.parametrize(
        "method", ("pxf", "logpxf", "cdf", "logcdf", "sf", "logsf", "ppf", "isf")
    )
    def test_eval_methods_array(self, data, wrapper, method):
        dist, method = get_dist_and_clean_method(wrapper, data, method)
        vals = np.linspace(0, 1, 10)
        meth = getattr(dist, method)
        with warnings.catch_warnings():
            if wrapper == "preliz" and method in ("logcdf", "logsf"):
                warnings.filterwarnings(
                    "ignore", message="divide by zero encountered in log", category=RuntimeWarning
                )
            out = meth(vals)
        assert out.ndim == 4
        if method in {"ppf", "isf"}:
            assert "quantile" in out.dims
        else:
            assert "point" in out.dims

    @pytest.mark.parametrize(
        "method", ("pxf", "logpxf", "cdf", "logcdf", "sf", "logsf", "ppf", "isf")
    )
    @pytest.mark.parametrize("in_type", ("scalar", "dataarray"))
    def test_eval_methods_dataarray(self, data, wrapper, method, in_type):
        dist, method = get_dist_and_clean_method(wrapper, data, method, in_type=in_type)
        meth = getattr(dist, method)
        with warnings.catch_warnings():
            if wrapper == "preliz" and method in ("logcdf", "logsf"):
                warnings.filterwarnings(
                    "ignore", message="divide by zero encountered in log", category=RuntimeWarning
                )
            out = meth(data["x_plot"])
        assert out.ndim == 4
        assert "plot_dim" in out.dims

    @pytest.mark.parametrize("dim_names", (None, ("name1", "name2")))
    @pytest.mark.parametrize("in_type", ("scalar", "dataarray"))
    def test_rv_method(self, data, wrapper, dim_names, in_type):
        dist = get_dist_and_clean_method(wrapper, data, in_type=in_type)
        out = dist.rvs(size=(2, 7), dims=dim_names)
        if dim_names is None:
            dim_names = ["rv_dim0", "rv_dim1"]
        assert_dims_in_da(out, dim_names)
        assert len(out[dim_names[0]]) == 2
        assert len(out[dim_names[1]]) == 7
        assert all(np.all(out.coords[name] == coords) for name, coords in data.coords.items())

    @pytest.mark.parametrize("in_type", ("scalar", "dataarray"))
    def test_rv_method_coords(self, data, wrapper, in_type):
        dist = get_dist_and_clean_method(wrapper, data, in_type=in_type)
        coords = {"new_dim": ["true", "false"], "extra_dim": list("abcdefg")}
        out = dist.rvs(coords=coords)
        assert_dims_in_da(out, list(coords))
        assert len(out["new_dim"]) == 2
        assert len(out["extra_dim"]) == 7
        assert np.all(out.coords["new_dim"] == np.array(coords["new_dim"]))
        assert np.all(out.coords["extra_dim"] == np.array(coords["extra_dim"]))
        assert all(np.all(out.coords[name] == coord) for name, coord in data.coords.items())

    @pytest.mark.parametrize("size", (1, 10))
    @pytest.mark.parametrize("dims", (None, "name", ["name"]))
    def test_rv_method_scalar_size(self, data, wrapper, size, dims):
        dist = get_dist_and_clean_method(wrapper, data)
        out = dist.rvs(size=size, dims=dims)
        dim_name = "rv_dim0" if dims is None else "name"
        if size == 1:
            assert dim_name not in out.dims
        else:
            assert dim_name in out.dims
            assert len(out[dim_name]) == size

    def test_non_broadcastable_input(self, data, wrapper):
        dist = get_dist_and_clean_method(wrapper, data, in_type="scalar")
        out = dist.cdf([1, 2])
        expected_dims = [*data["mu"].dims, "point"]
        assert out.ndim == len(expected_dims)
        assert_dims_in_da(out, expected_dims)

    def test_kwargs_input(self, data, wrapper):
        if wrapper == "rv_continuous":
            pytest.skip(
                "Only kwargs are valid for SciPy RVs, can't check match between args and kwargs"
            )
        if wrapper == "continuous":
            dist1 = XrContinuousRV(stats.norm, data["mu"], 1)
            dist2 = XrContinuousRV(stats.norm, loc=data["mu"], scale=1)
        elif wrapper == "preliz":
            try:
                from preliz import Normal

                dist1 = XrContinuousRV(Normal, data["mu"], 1)
                dist2 = XrContinuousRV(Normal, mu=data["mu"], sigma=1)
            except ImportError:
                pytest.skip("PreliZ not installed")
        elif wrapper == "discrete":
            dist1 = XrDiscreteRV(stats.poisson, data["mu"], 1)
            dist2 = XrDiscreteRV(stats.poisson, mu=data["mu"], loc=1)
        else:
            raise ValueError(f"Unkown value for 'wrapper': {wrapper}")
        out1 = dist1.cdf([1, 2])
        out2 = dist2.cdf([1, 2])
        expected_dims = [*data["mu"].dims, "point"]
        assert out2.ndim == len(expected_dims)
        assert_dims_in_da(out2, expected_dims)
        assert_allclose(out1, out2, atol=1e-15)


@pytest.mark.parametrize("vals", (([1, 2], [[1, 0.9], [0.9, 3]]), ([-3, 0], [[2, 0.3], [0.3, 2]])))
class TestMvNormal:
    """Test multivariate normal class.

    Tests are quite ad-hoc and empirical.
    """

    def test_rvs_method(self, vals):
        mean = xr.DataArray(vals[0], dims=["obs"])
        cov = xr.DataArray(vals[1], dims=["obs", "obs2"])
        dist = multivariate_normal(mean, cov, dims=("obs", "obs2"))
        samples = dist.rvs(random_state=3, size=(4, 10000), rv_dims=("chain", "draw"))
        assert_allclose(mean, samples.mean(("chain", "draw")), atol=1e-2, rtol=1e-2)
        assert_allclose(
            cov,
            xr.cov(samples, samples.rename(obs="obs2"), dim=("chain", "draw")),
            atol=0.1,
            rtol=1e-2,
        )

    def test_pdf_method(self, vals):
        mean = xr.DataArray(vals[0], dims=["obs"])
        cov = xr.DataArray(vals[1], dims=["obs", "obs2"])
        dist = multivariate_normal(mean, cov, dims=("obs", "obs2"))
        dist_sp = stats.multivariate_normal(mean, cov)
        points = [[0, 0], [1, 3], [-1, 2], [-3, -2]]
        x = xr.DataArray(points, dims=["point", "obs"])
        pdf_xr = dist.pdf(x)
        for i, point in enumerate(points):
            assert np.allclose(dist_sp.pdf(point), pdf_xr.isel(point=i))


@pytest.mark.parametrize("xr_obj", ("DataArray", "Dataset", "stacked"))
class TestStats:
    @pytest.mark.parametrize("dims", ("chain", ("chain", "draw"), None))
    def test_rankdata(self, data, dims, xr_obj):
        if xr_obj == "DataArray":
            xr_in = data["score"]
        elif xr_obj == "Dataset":
            if dims is None:
                pytest.skip("rankdata doesn't support Dataset input and dims=None")
            xr_in = data[["mu", "sigma"]]
        else:
            xr_in = (
                data["score"].rename(chain="aux").expand_dims(aux2=2).stack(chain=("aux", "aux2"))
            )
        xr_out = rankdata(xr_in, dims=dims)
        if xr_obj in ("DataArray", "stacked"):
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
    def test_reduce_function(self, data, stacked_data, dims, func, xr_obj):
        if xr_obj == "DataArray":
            xr_in = data["mu"]
        elif xr_obj == "Dataset":
            if dims is None and func is logsumexp:
                pytest.skip("logsumexp doesn't support Dataset input and dims=None")
            xr_in = data[["mu", "sigma"]]
        else:
            xr_in = stacked_data
        xr_out = func(xr_in, dims=dims)
        if xr_obj in ("DataArray", "stacked"):
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


def test_logsumexp_b(data):
    b_da = xr.DataArray([1, 2, 1, 1], dims=["chain"], coords={"chain": data.chain})
    out = logsumexp(data["mu"], dims="draw", b=b_da)
    out1 = logsumexp(data["mu"].sel(chain=0), dims="draw", b=1)
    out2 = logsumexp(data["mu"].sel(chain=1), dims="draw", b=2)
    assert_dims_in_da(out, ("chain", "team"))
    assert_dims_not_in_da(out, ["draw"])
    assert_allclose(out.sel(chain=0), out1)
    assert_allclose(out.sel(chain=1), out2)


def test_mad_da_scale(data):
    s_da = xr.DataArray([1, 2, 1, 1], dims=["chain"], coords={"chain": data.chain})
    out = median_abs_deviation(data["mu"], dims="draw", scale=s_da)
    out1 = median_abs_deviation(data["mu"].sel(chain=0), dims="draw", scale=1)
    out2 = median_abs_deviation(data["mu"].sel(chain=1), dims="draw", scale=2)
    assert_dims_in_da(out, ("chain", "team"))
    assert_dims_not_in_da(out, ["draw"])
    assert_allclose(out.sel(chain=0), out1)
    assert_allclose(out.sel(chain=1), out2)
