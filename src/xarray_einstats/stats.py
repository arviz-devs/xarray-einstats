# pylint: disable=too-few-public-methods
"""Wrappers for :mod:`scipy.stats` distributions.

.. note::

    These wrapper classes set some defaults and ensure
    proper alignment and broadcasting of all inputs, but
    use :func:`xarray.apply_ufunc` under the hood.
    This means that while using kwargs for distribution
    parameters is supported, using positional arguments
    is recommended. In fact, if no positional arguments
    are present, automatic broadcasting will still work
    but the output will be a numpy array.
"""

from collections.abc import Sequence

import numpy as np
import xarray as xr
from scipy import stats

__all__ = ["XrContinuousRV", "XrDiscreteRV", "rankdata", "gmean", "hmean"]


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
    def aux(self, *args, **kwargs):
        if args:
            x_or_q = args[0]
            if not isinstance(x_or_q, xr.DataArray):
                x_or_q = xr.DataArray(
                    np.asarray(x_or_q),
                    dims=["quantile"],
                    coords={"quantile": np.asarray(x_or_q)},
                )
                args = (x_or_q, *args[1:])
        args, kwargs = self._broadcast_args(args, kwargs)  # pylint: disable=protected-access
        meth = getattr(self.dist, method)
        return xr.apply_ufunc(meth, *args, kwargs=kwargs)

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
        len_args = len(self.args) + len(args)
        all_args = [*self.args, *args, *self.kwargs.values(), *kwargs.values()]
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
        all_keys = list(self.kwargs.keys()) + list(kwargs.keys())
        args = all_args[:len_args]
        kwargs = dict(zip(all_keys, all_args[len_args:]))
        return args, kwargs

    def rvs(self, *args, size=1, random_state=None, dims=None, **kwargs):
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

        if isinstance(size, (Sequence, np.ndarray)):
            if dims is None:
                dims = [f"rv_dim{i}" for i, _ in enumerate(size)]
            else:
                if len(dims) != len(size):
                    raise ValueError("dims and size must have the same length")
            size = (*size, *size_in)
        elif size > 1:
            if dims is None:
                dims = ["rv_dim0"]
            elif isinstance(dims, str):
                dims = [dims]
            if len(dims) != 1:
                raise ValueError("dims and size must have the same length")
            size = (size, *size_in)
        else:
            if size_in:
                size = size_in

        if dims is None:
            dims = tuple()

        return xr.apply_ufunc(
            self.dist.rvs,
            *args,
            kwargs={**kwargs, "size": size, "random_state": random_state},
            input_core_dims=[dims_in for _ in args],
            output_core_dims=[[*dims, *dims_in]],
        )


setattr(XrRV, "cdf", _wrap_method("cdf"))
setattr(XrRV, "logcdf", _wrap_method("logcdf"))
setattr(XrRV, "sf", _wrap_method("sf"))
setattr(XrRV, "logsf", _wrap_method("logsf"))
setattr(XrRV, "ppf", _wrap_method("ppf"))
setattr(XrRV, "isf", _wrap_method("isf"))


class XrContinuousRV(XrRV):
    """Wrapper for subclasses of :class:`~scipy.stats.rv_continuous`.

    See Also
    --------
    xarray_einstats.stats.XrDiscreteRV
    """


setattr(XrContinuousRV, "pdf", _wrap_method("pdf"))
setattr(XrContinuousRV, "logpdf", _wrap_method("logpdf"))


class XrDiscreteRV(XrRV):
    """Wrapper for subclasses of :class:`~scipy.stats.rv_discrete`.

    See Also
    --------
    xarray_einstats.stats.XrDiscreteRV
    """


setattr(XrDiscreteRV, "pmf", _wrap_method("pmf"))
setattr(XrDiscreteRV, "logpmf", _wrap_method("logpmf"))


def _add_docstrings(cls, wrapped_cls, methods):
    """Add one line docstrings to wrapper classes."""
    for method_name in methods:
        method = getattr(cls, method_name)
        setattr(
            method,
            "__doc__",
            f"Method wrapping :meth:`scipy.stats.{wrapped_cls}.{method_name}` "
            "with :func:`xarray.apply_ufunc`",
        )


base_methods = ["cdf", "logcdf", "sf", "logsf", "ppf", "isf", "rvs"]
_add_docstrings(XrContinuousRV, "rv_continuous", base_methods + ["pdf", "logpdf"])
_add_docstrings(XrDiscreteRV, "rv_discrete", base_methods + ["pmf", "logpmf"])


def _apply_nonreduce_func(func, da, dims, kwargs, func_kwargs=None):
    """Help wrap functions with a single input that return an output with the same size."""
    unstack = False

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
    """Wrap and extend :func:`scipy.stats.rankdata`."""
    rank_kwargs = {"axis": -1}
    if method is not None:
        rank_kwargs["method"] = method
    if dims is None:
        dims = get_default_dims(da.dims)
    return _apply_nonreduce_func(stats.rankdata, da, dims, kwargs, rank_kwargs)


def gmean(da, dims=None, dtype=None, weights=None, **kwargs):
    """Wrap and extend :func:`scipy.stats.gmean`."""
    gmean_kwargs = {"axis": -1}
    if dtype is not None:
        gmean_kwargs["dtype"] = dtype
    if weights is not None:
        gmean_kwargs["weights"] = weights
    if dims is None:
        dims = get_default_dims(da.dims)
    return _apply_reduce_func(stats.gmean, da, dims, kwargs, gmean_kwargs)


def hmean(da, dims=None, dtype=None, **kwargs):
    """Wrap and extend :func:`scipy.stats.hmean`."""
    hmean_kwargs = {"axis": -1}
    if dtype is not None:
        hmean_kwargs["dtype"] = dtype
    if dims is None:
        dims = get_default_dims(da.dims)
    return _apply_reduce_func(stats.hmean, da, dims, kwargs, hmean_kwargs)
