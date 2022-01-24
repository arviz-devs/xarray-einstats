"""Module with numba enhanced functions."""
import numba
import numpy as np
import xarray as xr

__all__ = ["histogram"]


@numba.guvectorize(
    [
        "void(uint8[:], uint8[:], uint8[:])",
        "void(uint16[:], uint16[:], uint16[:])",
        "void(uint32[:], uint32[:], uint32[:])",
        "void(uint64[:], uint64[:], uint64[:])",
        "void(int8[:], int8[:], int8[:])",
        "void(int16[:], int16[:], int16[:])",
        "void(int32[:], int32[:], int32[:])",
        "void(int64[:], int64[:], int64[:])",
        "void(float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:])",
    ],
    "(n),(m)->(m)",
    cache=True,
    target="parallel",
    nopython=True,
)
def hist_ufunc(data, bin_edges, res):
    """Use :func:`numba.guvectorize` to convert numpy histogram into a ufunc.

    Notes
    -----
    ``bin_edges`` is a required argument because it is needed to have a valid call
    signature. The shape of the output must be generated from the dimensions available
    in the inputs; they can be in different order, duplicated or reduced, but the output
    can't introduce new dimensions.
    """
    # TODO: check signatures
    m = len(bin_edges)
    res[:] = 0
    aux, _ = np.histogram(data, bins=bin_edges)
    for i in numba.prange(m - 1):
        res[i] = aux[i]


def histogram(da, dims, bins=None, density=False, **kwargs):
    """Numbify :func:`numpy.histogram` to suport vectorized histograms.

    Parameters
    ----------
    da : DataArray
        Data to be binned.
    dims : str or list of str
        Dimensions that should be reduced by binning.
    bins : array_like, int or str, optional
        Passed to :func:`numpy.histogram_bin_edges`. If ``None`` (the default)
        ``histogram_bin_edges`` is called without arguments. Bin edges
        are shared by all generated histograms.
    density : bool, optional
        If ``False``, the result will contain the number of samples in each bin.
        If ``True``, the result is the value of the probability density function at the bin,
        normalized such that the integral over the range is 1.
        Note that the sum of the histogram values will not be equal to 1 unless
        bins of unity width are chosen; it is not a probability mass function.
    kwargs : dict, optional
        Passed to :func:`xarray.apply_ufunc`

    Returns
    -------
    DataArray
        Returns a DataArray with the histogram results. The dimensions provided
        in ``dims`` are reduced into a ``bin`` dimension (without coordinates).
        The bin edges are returned as coordinate values indexed by the dimension
        ``bin``, left bin edges are stored as ``left_edges`` right ones as
        ``right_edges``.

    Examples
    --------
    Use `histogram` to compute multiple histograms in a vectorized fashion, binning
    along both `chain` and `draw` dimensions but not the `match` one. Consequently,
    `histogram` generates one independent histogram per `match`:

    .. jupyter-notebook::

        from xarray_einstats import tutorial, numba
        ds = tutorial.generate_mcmc_like_dataset(3)
        numba.tutorial(ds["score"])

    Note how the return is a single DataArray, not an array with the histogram and another
    with the bin edges. That is because the bin edges are included as coordinate values.

    """
    # TODO: make dask compatible even when bin edges are not passed manually
    if bins is None:
        bin_edges = np.histogram_bin_edges(da.values.flatten())
    elif isinstance(bins, (str, int)):
        bin_edges = np.histogram_bin_edges(da.values.flatten(), bins=bin_edges)
    else:
        bin_edges = bins
    if not isinstance(dims, str):
        da = da.stack(__hist__=dims)
        dims = "__hist__"
    histograms = xr.apply_ufunc(
        hist_ufunc,
        da,
        bin_edges,
        input_core_dims=[[dims], ["bin"]],
        output_core_dims=[["bin"]],
        **kwargs
    )
    histograms = histograms.isel({"bin": slice(None, -1)}).assign_coords(
        left_edges=("bin", bin_edges[:-1]), right_edges=("bin", bin_edges[1:])
    )
    if density:
        return histograms / (
            histograms.sum("bin") * (histograms.right_edges - histograms.left_edges)
        )
    return histograms
