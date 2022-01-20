"""Tutorial module with data for docs and quick testing."""
import numpy as np
import xarray as xr


def generate_mcmc_like_dataset(seed=None):
    """Generate a Dataset with multiple variables, some with dimensions from mcmc sampling."""
    rng = np.random.default_rng(seed)
    ds = xr.Dataset(
        {
            "x_plot": (["plot_dim"], np.linspace(0, 10, 20)),
            "mu": (["chain", "draw", "team"], rng.exponential(size=(4, 10, 6))),
            "sigma": (["chain", "draw"], rng.exponential(size=(4, 10))),
            "score": (["chain", "draw", "match"], rng.poisson(size=(4, 10, 12))),
        },
        coords={"team": list("abcdef"), "chain": np.arange(4), "draw": np.arange(10)},
    )
    return ds


def generate_matrices_dataarray(seed=None):
    """Generate a 4d DataArray representing a batch of matrices."""
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.exponential(size=(10, 3, 4, 4)), dims=["batch", "experiment", "dim", "dim2"]
    )
