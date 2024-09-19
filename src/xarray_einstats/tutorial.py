"""Tutorial module with data for docs and quick testing."""

import numpy as np
import xarray as xr

try:
    from IPython import get_ipython
    from PIL.Image import fromarray

    array_display = True  # pylint: disable=invalid-name
except ModuleNotFoundError:
    array_display = False  # pylint: disable=invalid-name


def generate_mcmc_like_dataset(seed=None):
    """Generate a Dataset with multiple variables, some with dimensions from mcmc sampling.

    Parameters
    ----------
    seed : int or sequence of int, optional
        The random seed used to initialize :func:`numpy.random.default_rng`.

    Examples
    --------
    The dataset generated is the following:

    .. jupyter-execute::

        from xarray_einstats import tutorial
        tutorial.generate_mcmc_like_dataset(3)

    Notes
    -----
    This function is not part of the public API and is designed for use in our documentation.
    In addition to generating the data, it also sets ``display_expand_data=False`` to
    avoid taking too much virtual space with examples.

    """
    xr.set_options(display_expand_data=False)
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
    """Generate a 4d DataArray representing a batch of matrices.

    Parameters
    ----------
    seed : int or sequence of int, optional
        The random seed used to initialize :func:`numpy.random.default_rng`.

    Examples
    --------
    The dataset generated is the following:

    .. jupyter-execute::

        from xarray_einstats import tutorial
        tutorial.generate_matrices_dataarray(5)

    Notes
    -----
    This function is not part of the public API and is designed for use in our documentation.
    In addition to generating the data, it also sets ``display_expand_data=False`` to
    avoid taking too much virtual space with examples.

    """
    xr.set_options(display_expand_data=False)
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.exponential(size=(10, 3, 4, 4)), dims=["batch", "experiment", "dim", "dim2"]
    )


if array_display:

    def display_np_arrays_as_images():
        """Display numpy arrays as images by default in IPython and Jupyter.

        Only needs to be imported for the behaviour to be configured.
        """
        # pylint: disable=protected-access

        def np_to_png(a):
            if 2 <= len(a.shape) <= 3:
                return fromarray(np.array(np.clip(a, 0, 1) * 255, dtype="uint8"))._repr_png_()
            return fromarray(np.zeros([1, 1], dtype="uint8"))._repr_png_()

        def np_to_text(obj, p, cycle):  # pylint: disable=unused-argument, invalid-name
            if len(obj.shape) < 2:
                print(repr(obj))
            if 2 <= len(obj.shape) <= 3:
                pass
            else:
                print(f"<array of shape {obj.shape}>")

        get_ipython().display_formatter.formatters["image/png"].for_type(np.ndarray, np_to_png)
        get_ipython().display_formatter.formatters["text/plain"].for_type(np.ndarray, np_to_text)
