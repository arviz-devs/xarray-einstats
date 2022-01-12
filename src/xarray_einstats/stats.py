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
        args, kwargs = self._broadcast_args(  # pylint: disable=protected-access
            args, kwargs
        )
        meth = getattr(self.dist, method)
        return xr.apply_ufunc(meth, *args, kwargs=kwargs)

    return aux


class XrRV:
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
    """Wrapper for subclasses of :class:`~scipy.stats.rv_continuous`."""


setattr(XrContinuousRV, "pdf", _wrap_method("pdf"))
setattr(XrContinuousRV, "logpdf", _wrap_method("logpdf"))


class XrDiscreteRV(XrRV):
    """Wrapper for subclasses of :class:`~scipy.stats.rv_discrete`."""


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
