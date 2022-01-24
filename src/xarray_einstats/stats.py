# pylint: disable=too-few-public-methods
"""Wrappers for :mod:`scipy.stats` distributions."""

from collections.abc import Sequence

import numpy as np
import xarray as xr
from scipy import stats

__all__ = [
    "XrContinuousRV",
    "XrDiscreteRV",
    "rankdata",
    "gmean",
    "hmean",
    "circmean",
    "circvar",
    "circstd",
]


def get_default_dims(dims):
    """Get default dims on which to perfom an operation.

    Whenever a function from :mod:`xarray_einstats.stats` is called with
    ``dims=None`` (the default) this function is called to choose the
    default dims on which to operate out of the list with all the dims present.

    This function is thought to be monkeypatched by domain specific applications
    as shown in the examples.

    Parameters
    ----------
    dims : list of str
        List with all the dimensions of the input DataArray in the order they
        appear.

    Returns
    -------
    list of str
        List with the dimensions on which to apply the operation.
        ``xarray_einstats`` defaults to applying the operation to all
        dimensions. Monkeypatch this function to get a different result.

    Examples
    --------
    The ``xarray_einstats`` default behaviour is operating (averaging in this case)
    over all dimensions present in the input DataArray:

    .. jupyter-execute::

        from xarray_einstats import stats, tutorial
        da = tutorial.generate_mcmc_like_dataset(3)["mu"]
        stats.hmean(da)

    Here we show how to monkeypatch ``get_default_dims`` to get a different default
    behaviour. If you use ``xarray_einstats`` and {doc}`arviz:index` to work
    with MCMC results, operating over chain and dim only might be a better default:

    .. jupyter-execute::

        def get_default_dims(dims):
            out = [dim for dim in ("chain", "draw") if dim in dims]
            if not out:  # if chain nor draw are present fall back to all dims
                return dims
            return out
        stats.get_default_dims = get_default_dims
        stats.hmean(da)

    You can still use ``dims`` explicitly to average over any custom dimension

    .. jupyter-execute::

        stats.hmean(da, dims="team")

    """
    return dims


def _wrap_method(method):
    def aux(self, *args, apply_kwargs=None, **kwargs):
        if apply_kwargs is None:
            apply_kwargs = {}
        meth = getattr(self.dist, method)
        if args:
            x_or_q = args[0]
            dim_name = "quantile" if method in {"ppf", "isf"} else "point"
            if not isinstance(x_or_q, xr.DataArray):
                x_or_q = xr.DataArray(
                    np.asarray(x_or_q),
                    dims=[dim_name],
                    coords={dim_name: np.asarray(x_or_q)},
                )
                args = (x_or_q, *args[1:])
        args, kwargs = self._broadcast_args(args, kwargs)  # pylint: disable=protected-access
        return xr.apply_ufunc(meth, *args, kwargs=kwargs, **apply_kwargs)

    return aux


class XrRV:
    """Base random variable wrapper class.

    Most methods have a common signature between continuous and
    discrete variables in scipy. We define a base wrapper and
    then subclass it to add the specific methods like pdf or pmf.

    Notes
    -----
    One of the main goals of this library is ease of maintenance.
    We could wrap each distribution to preserve call signatures
    and avoid different behaviour between passing input arrays
    as args or kwargs, but so far we don't consider what we'd won
    doing this to be worth the extra maintenance burden.
    """

    def __init__(self, dist, *args, **kwargs):
        self.dist = dist
        self.args = args
        self.kwargs = kwargs

    def _broadcast_args(self, args, kwargs):
        len_args = len(args) + len(self.args)
        all_args = [*args, *self.args, *kwargs.values(), *self.kwargs.values()]
        broadcastable = []
        non_broadcastable = []
        b_idx = []
        n_idx = []
        for i, a in enumerate(all_args):
            if isinstance(a, xr.DataArray):
                broadcastable.append(a)
                b_idx.append(i)
            else:
                non_broadcastable.append(a)
                n_idx.append(i)
        broadcasted = list(xr.broadcast(*broadcastable))
        all_args = [
            x
            for x, _ in sorted(
                zip(broadcasted + non_broadcastable, b_idx + n_idx),
                key=lambda pair: pair[1],
            )
        ]
        all_keys = list(kwargs.keys()) + list(self.kwargs.keys())
        args = all_args[:len_args]
        kwargs = dict(zip(all_keys, all_args[len_args:]))
        return args, kwargs

    def rvs(self, *args, size=1, random_state=None, dims=None, apply_kwargs=None, **kwargs):
        """Implement base rvs method.

        In scipy, rvs has a common signature that doesn't depend on continuous
        or discrete, so we can define it here.
        """
        args, kwargs = self._broadcast_args(args, kwargs)
        size_in = tuple()
        dims_in = tuple()
        for a in (*args, *kwargs.values()):
            if isinstance(a, xr.DataArray):
                size_in = a.shape
                dims_in = a.dims
                break

        if isinstance(dims, str):
            dims = [dims]

        if isinstance(size, (Sequence, np.ndarray)):
            if dims is None:
                dims = [f"rv_dim{i}" for i, _ in enumerate(size)]
            if len(dims) != len(size):
                raise ValueError("dims and size must have the same length")
            size = (*size, *size_in)
        elif size > 1:
            if dims is None:
                dims = ["rv_dim0"]
            if len(dims) != 1:
                raise ValueError("dims and size must have the same length")
            size = (size, *size_in)
        else:
            if size_in:
                size = size_in
            dims = None

        if dims is None:
            dims = tuple()

        if apply_kwargs is None:
            apply_kwargs = {}

        return xr.apply_ufunc(
            self.dist.rvs,
            *args,
            kwargs={**kwargs, "size": size, "random_state": random_state},
            input_core_dims=[dims_in for _ in args],
            output_core_dims=[[*dims, *dims_in]],
            **apply_kwargs,
        )


class XrContinuousRV(XrRV):
    """Wrapper for subclasses of :class:`~scipy.stats.rv_continuous`.

    Usage examples available at :ref:`stats_tutorial`

    See Also
    --------
    xarray_einstats.stats.XrDiscreteRV

    Examples
    --------
    Evaluate the ppf of a Student-T distribution from DataArrays that need
    broadcasting:

    .. jupyter-execute::

        from xarray_einstats import tutorial
        from xarray_einstats.stats import XrContinuousRV
        from scipy import stats
        ds = tutorial.generate_mcmc_like_dataset(3)
        dist = XrContinuousRV(stats.t, 3, ds["mu"], ds["sigma"])
        dist.ppf([.1, .5, .6])

    """


class XrDiscreteRV(XrRV):
    """Wrapper for subclasses of :class:`~scipy.stats.rv_discrete`.

    Usage examples available at :ref:`stats_tutorial`

    See Also
    --------
    xarray_einstats.stats.XrDiscreteRV

    Examples
    --------
    Evaluate the ppf of a Student-T distribution from DataArrays that need
    broadcasting:

    .. jupyter-execute::

        from xarray_einstats import tutorial
        from xarray_einstats.stats import XrDiscreteRV
        from scipy import stats
        ds = tutorial.generate_mcmc_like_dataset(3)
        dist = XrDiscreteRV(stats.poisson, ds["mu"])
        dist.ppf([.1, .5, .6])

    """


def _add_documented_method(cls, wrapped_cls, methods, extra_docs=None):
    """Register methods to XrRV classes and document them from a template."""
    if extra_docs is None:
        extra_docs = {}
    for method_name in methods:
        extra_doc = extra_docs.get(method_name, "")
        if method_name == "rvs":
            if wrapped_cls == "rv_generic":
                continue
            method = cls.rvs
        else:
            method = _wrap_method(method_name)
        setattr(
            method,
            "__doc__",
            f"Method wrapping :meth:`scipy.stats.{wrapped_cls}.{method_name}` "
            "with :func:`xarray.apply_ufunc`\n\nUsage examples available at "
            f":ref:`stats_tutorial/dists`.\n\n{extra_doc}",
        )
        setattr(cls, method_name, method)


doc_extras = dict(
    rvs="""
Parameters
----------
args : scalar or array_like, optional
    Passed to the scipy distribution after broadcasting.
size : int of sequence of ints, optional
    The number of samples to draw *per array element*. If the distribution
    parameters broadcast to a ``(4, 10, 6)`` shape and ``size=(5, 3)`` then
    the output shape is ``(5, 3, 4, 10, 6)``. This differs from the scipy
    implementation. Here, all broadcasting and alignment is done for you,
    you give the dimensions the right names, and broadcasting just happens.
    If ``size`` followed scipy behaviour, you'd be forced to broadcast
    to provide a valid value which would defeat the ``xarray_einstats`` goal
    of handling all alignment and broadcasting for you.
random_state : optional
    Passed as is to the wrapped scipy distribution
dims : sequence of str, optional
    Dimension names for the dimensions created due to ``size``. If present
    it must have the same length as ``size``.
apply_kwargs : dict, optional
    Passed to :func:`xarray.apply_ufunc`
kwargs : dict, optional
    Passed to the scipy distribution after broadcasting using the same key.
"""
)
base_methods = ["cdf", "logcdf", "sf", "logsf", "ppf", "isf", "rvs"]
_add_documented_method(XrRV, "rv_generic", base_methods, doc_extras)
_add_documented_method(
    XrContinuousRV, "rv_continuous", base_methods + ["pdf", "logpdf"], doc_extras
)
_add_documented_method(XrDiscreteRV, "rv_discrete", base_methods + ["pmf", "logpmf"], doc_extras)


def _apply_nonreduce_func(func, da, dims, kwargs, func_kwargs=None):
    """Help wrap functions with a single input that return an output with the same size."""
    unstack = False

    if dims is None:
        dims = get_default_dims(da.dims)
    if not isinstance(dims, str):
        da = da.stack(__aux_dim__=dims)
        core_dims = ["__aux_dim__"]
        unstack = True
    else:
        core_dims = [dims]
    out_da = xr.apply_ufunc(
        func,
        da,
        input_core_dims=[core_dims],
        output_core_dims=[core_dims],
        kwargs=func_kwargs,
        **kwargs,
    )
    if unstack:
        return out_da.unstack("__aux_dim__")
    return out_da


def _apply_reduce_func(func, da, dims, kwargs, func_kwargs=None):
    """Help wrap functions with a single input that return an output after reducing some dimensions.

    It assumes that the function to be applied only takes ``int`` as ``axis`` and stacks multiple
    dimensions if necessary to support reducing multiple dimensions at once.
    """
    if dims is None:
        dims = get_default_dims(da.dims)
    if not isinstance(dims, str):
        da = da.stack(__aux_dim__=dims)
        core_dims = ["__aux_dim__"]
    else:
        core_dims = [dims]
    out_da = xr.apply_ufunc(
        func, da, input_core_dims=[core_dims], output_core_dims=[[]], kwargs=func_kwargs, **kwargs
    )
    return out_da


def rankdata(da, dims=None, method=None, **kwargs):
    """Wrap and extend :func:`scipy.stats.rankdata`.

    Usage examples available at :ref:`stats_tutorial`

    See Also
    --------
    xarray.DataArray.rank : Similar function but without a ``method`` argument available.
    """
    rank_kwargs = {"axis": -1}
    if method is not None:
        rank_kwargs["method"] = method
    return _apply_nonreduce_func(stats.rankdata, da, dims, kwargs, rank_kwargs)


def gmean(da, dims=None, dtype=None, weights=None, **kwargs):
    """Wrap and extend :func:`scipy.stats.gmean`.

    Usage examples available at :ref:`stats_tutorial`
    """
    gmean_kwargs = {"axis": -1}
    if dtype is not None:
        gmean_kwargs["dtype"] = dtype
    if weights is not None:
        gmean_kwargs["weights"] = weights
    return _apply_reduce_func(stats.gmean, da, dims, kwargs, gmean_kwargs)


def hmean(da, dims=None, dtype=None, **kwargs):
    """Wrap and extend :func:`scipy.stats.hmean`.

    Usage examples available at :ref:`stats_tutorial`
    """
    hmean_kwargs = {"axis": -1}
    if dtype is not None:
        hmean_kwargs["dtype"] = dtype
    return _apply_reduce_func(stats.hmean, da, dims, kwargs, hmean_kwargs)


def circmean(da, high=2 * np.pi, low=0, dims=None, nan_policy=None, **kwargs):
    """Wrap and extend :func:`scipy.stats.circmean`.

    Usage examples available at :ref:`stats_tutorial`
    """
    circmean_kwargs = dict(axis=-1, high=high, low=low)
    if nan_policy is not None:
        circmean_kwargs["nan_policy"] = nan_policy
    return _apply_reduce_func(stats.circmean, da, dims, kwargs, circmean_kwargs)


def circvar(da, high=2 * np.pi, low=0, dims=None, nan_policy=None, **kwargs):
    """Wrap and extend :func:`scipy.stats.circvar`.

    Usage examples available at :ref:`stats_tutorial`
    """
    circvar_kwargs = dict(axis=-1, high=high, low=low)
    if nan_policy is not None:
        circvar_kwargs["nan_policy"] = nan_policy
    return _apply_reduce_func(stats.circvar, da, dims, kwargs, circvar_kwargs)


def circstd(da, high=2 * np.pi, low=0, dims=None, nan_policy=None, **kwargs):
    """Wrap and extend :func:`scipy.stats.circstd`.

    Usage examples available at :ref:`stats_tutorial`
    """
    circstd_kwargs = dict(axis=-1, high=high, low=low)
    if nan_policy is not None:
        circstd_kwargs["nan_policy"] = nan_policy
    return _apply_reduce_func(stats.circstd, da, dims, kwargs, circstd_kwargs)
