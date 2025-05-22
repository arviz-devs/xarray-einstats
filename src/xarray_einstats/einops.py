"""Wrappers for `einops <https://einops.rocks/>`_.

The einops module is available only from ``xarray_einstats.einops`` and is not
imported when doing ``import xarray_einstats``.
To use it you need to have installed einops manually or alternatively
install this library as ``xarray-einstats[einops]`` or ``xarray-einstats[all]``.
Details about the exact command are available at :ref:`installation`

.. tip::

    The functions above are also available via the ``.einops`` accessor
    from :class:`DataArray` objects. See :ref:`einops_tutorial` for
    example usage.

"""

from collections.abc import Hashable

import einops
import xarray as xr

__all__ = ["rearrange", "reduce", "DaskBackend"]


class DimHandler:
    """Handle converting actual dimension names to placeholders for einops."""

    def __init__(self):
        self.mapping = {}

    def get_name(self, dim):
        """Return or generate a placeholder for a dimension name."""
        if dim in self.mapping:
            return self.mapping.get(dim)
        dim_txt = f"d{len(self.mapping)}"
        self.mapping[dim] = dim_txt
        return dim_txt

    def get_names(self, dim_list):
        """Automate calling get_name with an iterable."""
        return " ".join((self.get_name(dim) for dim in dim_list))

    def rename_kwarg(self, key):
        """Process kwargs for axes_lengths.

        Users use as keys the dimension names they used in the input expressions
        which need to be converted and use the placeholder as key when passed
        to einops functions.
        """
        return self.mapping.get(key, key)


def process_pattern_list(redims, handler, allow_dict=True, allow_list=True):
    """Process a pattern list and convert it to an einops expression using placeholders.

    Parameters
    ----------
    redims : list of (hashable or list or dict)
        One of ``out_dims`` or ``in_dims`` in {func}`~xarray_einstats.einops.rearrange`
        or {func}`~xarray_einstats.einops.reduce`.
    handler : DimHandler
    allow_dict, allow_list : bool, optional
        Whether or not to allow lists or dicts as elements of ``redims``.
        When processing ``in_dims`` for example we need the names of
        the variables to be decomposed so dicts are required and lists
        are not accepted.

    Returns
    -------
    expression_dims : list of str
        A list with the names of the dimensions present in the out expression
    output_dims : list of str
        A list with the names of the dimensions present in the output.
        It differs from ``expression_dims`` because there might be dimensions
        being stacked.
    pattern : str
        The einops expression equivalent to the operations in `redims` pattern
        list.

    Examples
    --------
    Whenever we have groupings of dimensions (be it to decompose or to stack),
    ``expression_dims`` and ``output_dims`` differ:

    .. jupyter-execute::

        from xarray_einstats.einops import process_pattern_list, DimHandler
        handler = DimHandler()
        process_pattern_list(["a", {"b": ["c", "d"]}, ["e", "f", "g"]], handler)

    """
    out = []
    out_names = []
    txt = []
    for subitem in redims:
        if isinstance(subitem, Hashable):
            out.append(subitem)
            out_names.append(subitem)
            txt.append(handler.get_name(subitem))
        elif isinstance(subitem, dict) and allow_dict:
            if len(subitem) != 1:
                raise ValueError(
                    "dicts in pattern list must have a single key but instead "
                    f"found {len(subitem)}: {subitem.keys()}"
                )
            key, values = list(subitem.items())[0]
            if isinstance(values, Hashable):
                raise ValueError(
                    "Found values of hashable type in a pattern dict, use xarray.rename"
                )
            out.extend(values)
            out_names.append(key)
            txt.append(f"( {handler.get_names(values)} )")
        elif allow_list:
            out.extend(subitem)
            out_names.append("-".join(subitem))
            txt.append(f"( {handler.get_names(subitem)} )")
        else:
            raise ValueError(
                f"Found unsupported pattern type: {type(subitem)}, double check the docs. "
                "This could be for example is using lists/tuples as elements of in_dims argument"
            )
    return out, out_names, " ".join(txt)


def translate_pattern(pattern):
    """Translate a string pattern to a list pattern.

    Parameters
    ----------
    pattern : str
        Input pattern as a string.

    Returns
    -------
    list of (hashable or list or dict)
        Pattern translated to list, as used by the direct feature-full wrappers.

    Examples
    --------
    .. jupyter-execute::

        from xarray_einstats.einops import translate_pattern
        translate_pattern("a (c d)=b (e f g)")

    """
    dims = []
    current_dim = ""
    current_block = []
    parsing_block = 0  # 0=no block, 1=block, 2=just closed, waiting for key
    parsing_key = False
    for char in pattern.strip() + " ":
        if char == " ":
            if parsing_key:
                if current_dim:
                    dims.append({current_dim: current_block})
                else:
                    dims.append(current_block)
                current_block = []
                parsing_key = False
                parsing_block = False
            elif not current_dim:
                continue
            elif parsing_block:
                current_block.append(current_dim)
            else:
                dims.append(current_dim)
            current_dim = ""
        elif char == ")":
            if parsing_block:
                parsing_block = False
                parsing_key = True
                if current_dim:
                    current_block.append(current_dim)
                current_dim = ""
            else:
                raise ValueError("unmatched parenthesis")
        elif char == "(":
            parsing_block = 1
        elif char == "=":
            if not parsing_key:
                raise ValueError("= sign must follow a closing parenthesis )")
        else:
            current_dim += char
    return dims


def _rearrange(da, out_dims, in_dims=None, dim_lengths=None):
    """Wrap `einops.rearrange <https://einops.rocks/api/rearrange/>`_.

    This is the function that actually interfaces with ``einops``.
    :func:`xarray_einstats.einops.rearrange` is the user facing
    version as it exposes two possible APIs, one of them significantly
    less verbose and more friendly (but much less flexible).

    Parameters
    ----------
    da : xarray.DataArray
        Input DataArray to be rearranged
    out_dims : list of (str or list or dict)
        See docstring of :func:`~xarray_einstats.einops.rearrange`
    in_dims : list of (str or dict), optional
        See docstring of :func:`~xarray_einstats.einops.rearrange`
    dim_lengths : dict, optional
        kwargs with key equal to dimension names in ``out_dims``
        (that is, strings or dict keys) are passed to einops.rearrange
        the rest of keys are passed to :func:`xarray.apply_ufunc`
    """
    if dim_lengths is None:
        dim_lengths = {}

    da_dims = da.dims

    handler = DimHandler()
    if in_dims is None:
        in_dims = []
        in_names = []
        in_pattern = ""
    else:
        in_dims, in_names, in_pattern = process_pattern_list(
            in_dims, handler=handler, allow_list=False
        )
    # note, not using sets for da_dims to avoid transpositions on missing variables,
    # if they wanted to transpose those they would not be missing variables
    out_dims, out_names, out_pattern = process_pattern_list(out_dims, handler=handler)
    missing_in_dims = [dim for dim in da_dims if dim not in in_names]
    expected_missing = set(out_dims).union(in_names).difference(in_dims)
    missing_out_dims = [dim for dim in da_dims if dim not in expected_missing]

    # avoid using dimensions as core dims unnecesarly
    non_core_dims = [dim for dim in missing_in_dims if dim in missing_out_dims]
    missing_in_dims = [dim for dim in missing_in_dims if dim not in non_core_dims]
    missing_out_dims = [dim for dim in missing_out_dims if dim not in non_core_dims]

    non_core_pattern = handler.get_names(non_core_dims)
    pattern = f"{non_core_pattern} {handler.get_names(missing_in_dims)} {in_pattern} ->\
        {non_core_pattern} {handler.get_names(missing_out_dims)} {out_pattern}"

    axes_lengths = {
        handler.rename_kwarg(k): v for k, v in dim_lengths.items() if k in out_names + out_dims
    }
    kwargs = {k: v for k, v in dim_lengths.items() if k not in out_names + out_dims}
    return xr.apply_ufunc(
        einops.rearrange,
        da,
        pattern,
        input_core_dims=[missing_in_dims + in_names, []],
        output_core_dims=[missing_out_dims + out_names],
        kwargs=axes_lengths,
        **kwargs,
    )


def rearrange(da, pattern, pattern_in=None, dim_lengths=None, **dim_lengths_kwargs):
    """Expose `einops.rearrange <https://einops.rocks/api/rearrange/>`_ with an xarray-like API.

    It has two possible syntaxes which are independent and somewhat complementary.

    Wrapper around einops.rearrange with a very similar syntax.
    Spaces, parenthesis ``()`` and `->` are not allowed in dimension names.

    Parameters
    ----------
    da : xarray.DataArray
        Input array
    pattern : str or list of (hashable or list or dict)
        If `pattern` is a string, it uses the same syntax as einops
        with two caveats:

        * Unless splitting or stacking, you must use the actual dimension names.
        * When splitting or stacking you can use ``(dim1 dim2)=dim``. This is *necessary*
          for the left hand side as it identifies the dimension to split, and
          optional on the right hand side, if omitted the stacked dimension will be given
          a default name.

        If `pattern` is not a string, then it must be a list where each of its elements
        is one of: :term:`python:hashable`, ``list`` (to stack those dimensions and
        give them an arbitrary name) or ``dict`` (to stack the dimensions indicated
        as values of the dictionary and name the resulting dimensions with the key).

        `pattern` is then interpreted as the output side of the einops pattern.
        See :ref:`about_einops` for more details.
    pattern_in : list of (hashable or dict), optional
        The input pattern for the dimensions. It can only be provided if `pattern`
        is a ``list``. Also, note this is only necessary if you want to split some dimensions.

        The syntax and interpretation is the same as the case when `pattern` is a list,
        with the only difference that ``list`` elements are not allowed, the same way
        that ``(dim1 dim2)=dim`` is required on the left hand side when using string
        patterns.
    dim_lengths : dict of {hashable : int}, optional
        If the keys are dimensions present in `pattern` they will be passed to
        `einops.rearrange <https://einops.rocks/api/rearrange/>`_, otherwise,
        they are passed to :func:`xarray.apply_ufunc`.
    **dim_lengths_kwargs : int, optional
        kwarg version of `dim_lengths`.

    Returns
    -------
    xarray.DataArray

    See Also
    --------
    xarray_einstats.einops.reduce
    """
    if dim_lengths is None:
        dim_lengths = {}
    dim_lengths = {**dim_lengths, **dim_lengths_kwargs}
    if isinstance(pattern, str):
        if "->" in pattern:
            in_pattern, out_pattern = pattern.split("->")
            in_dims = translate_pattern(in_pattern)
        else:
            out_pattern = pattern
            in_dims = None
        out_dims = translate_pattern(out_pattern)
        return _rearrange(da, out_dims=out_dims, in_dims=in_dims, dim_lengths=dim_lengths)
    return _rearrange(da, out_dims=pattern, in_dims=pattern_in, dim_lengths=dim_lengths)


def _reduce(da, reduction, out_dims, in_dims=None, dim_lengths=None):
    """Wrap `einops.reduce <https://einops.rocks/api/reduce/>`_.

    This is the function that actually interfaces with ``einops``.
    :func:`xarray_einstats.einops.rearrange` is the user facing
    version as it exposes two possible APIs, one of them significantly
    less verbose and more friendly (but much less flexible).

    Parameters
    ----------
    da : xarray.DataArray
        Input DataArray to be reduced
    reduction : str or callable
        One of available reductions ('min', 'max', 'sum', 'mean', 'prod') by ``einops.reduce``,
        case-sensitive. Alternatively, a callable ``f(tensor, reduced_axes) -> tensor``
        can be provided. ``reduced_axes`` are passed as a list of int.
    out_dims : list of (str or list or dict)
        The output pattern for the dimensions.
        The dimensions present in
    in_dims : list of (str or dict), optional
        The input pattern for the dimensions.
        This is only necessary if you want to split some dimensions.
    dim_lengths : dict of {hashable : int}, optional
        kwargs with key equal to dimension names in ``out_dims``
        (that is, strings or dict keys) are passed to einops.rearrange
        the rest of keys are passed to :func:`xarray.apply_ufunc`
    """
    if dim_lengths is None:
        dim_lengths = {}

    da_dims = da.dims

    handler = DimHandler()
    if in_dims is None:
        in_dims = []
        in_names = []
        in_pattern = ""
    else:
        in_dims, in_names, in_pattern = process_pattern_list(
            in_dims, handler=handler, allow_list=False
        )
    # note, not using sets for da_dims to avoid transpositions on missing variables,
    # if they wanted to transpose those they would not be missing variables
    out_dims, out_names, out_pattern = process_pattern_list(out_dims, handler=handler)
    missing_in_dims = [dim for dim in da_dims if dim not in in_names]
    pattern = f"{handler.get_names(missing_in_dims)} {in_pattern} -> {out_pattern}"

    all_dims = set(out_dims + out_names + in_names + in_dims)
    axes_lengths = {handler.rename_kwarg(k): v for k, v in dim_lengths.items() if k in all_dims}
    kwargs = {k: v for k, v in dim_lengths.items() if k not in all_dims}
    return xr.apply_ufunc(
        einops.reduce,
        da,
        pattern,
        reduction,
        input_core_dims=[missing_in_dims + in_names, [], []],
        output_core_dims=[out_names],
        kwargs=axes_lengths,
        **kwargs,
    )


def reduce(da, pattern, reduction, pattern_in=None, dim_lengths=None, **dim_lengths_kwargs):
    """Expose `einops.reduce <https://einops.rocks/api/reduce/>`_ with an xarray-like API.

    It has two possible syntaxes which are independent and somewhat complementary.

    Parameters
    ----------
    da : xarray.DataArray
        Input array
    pattern : str or list of (str or list or dict)
        If `pattern` is a string, it uses the same syntax as einops
        with two caveats:

        * Unless splitting or stacking, you must use the actual dimension names.
        * When splitting or stacking you can use ``(dim1 dim2)=dim``. This is *necessary*
          for the left hand side as it identifies the dimension to split, and
          optional on the right hand side, if omitted the stacked dimension will be given
          a default name.

        If `pattern` is not a string, then it must be a list where each of its elements
        is one of: ``str``, ``list`` (to stack those dimensions and give them an
        arbitrary name) or ``dict of {str: list}`` (to stack the dimensions indicated
        as values of the dictionary and name the resulting dimensions with the key).

        `pattern` is then interpreted as the output side of the einops pattern. See
        TODO for more details.
    reduction : str or callable
        One of available reductions ('min', 'max', 'sum', 'mean', 'prod') by ``einops.reduce``,
        case-sensitive. Alternatively, a callable ``f(tensor, reduced_axes) -> tensor``
        can be provided. ``reduced_axes`` are passed as a list of int.
    pattern_in : list of (str or dict), optional
        The input pattern for the dimensions. It can only be provided if `pattern`
        is a ``list``. Also, note this is only necessary if you want to split some dimensions.

        The syntax and interpretation is the same as the case when `pattern` is a list,
        with the only difference that ``list`` elements are not allowed, the same way
        that ``(dim1 dim2)=dim`` is required on the left hand side when using string
    dim_lengths : dict of {hashable : int}, optional
        If the keys are dimensions present in `pattern` they will be passed to
        `einops.reduce <https://einops.rocks/api/reduce/>`_, otherwise,
        they are passed to :func:`xarray.apply_ufunc`.
    **dim_lengths_kwargs : int, optional
        Kwargs version of `dim_lengths`.

    Returns
    -------
    xarray.DataArray

    See Also
    --------
    xarray_einstats.einops.rearrange
    """
    if dim_lengths is None:
        dim_lengths = {}
    dim_lengths = {**dim_lengths, **dim_lengths_kwargs}
    if isinstance(pattern, str):
        if "->" in pattern:
            in_pattern, out_pattern = pattern.split("->")
            in_dims = translate_pattern(in_pattern)
        else:
            out_pattern = pattern
            in_dims = None
        out_dims = translate_pattern(out_pattern)
        return _reduce(da, reduction, out_dims=out_dims, in_dims=in_dims, dim_lengths=dim_lengths)
    return _reduce(da, reduction, out_dims=pattern, in_dims=pattern_in, dim_lengths=dim_lengths)


class DaskBackend(einops._backends.AbstractBackend):  # pylint: disable=protected-access
    """Dask backend class for einops.

    It should be imported before using functions of :mod:`xarray_einstats.einops`
    on Dask backed DataArrays.
    It doesn't need to be initialized or used explicitly

    Notes
    -----
    Class created from the advise on
    `issue einops#120 <https://github.com/arogozhnikov/einops/issues/120>`_ about Dask support.
    And from reading
    `einops/_backends <https://github.com/arogozhnikov/einops/blob/master/einops/_backends.py>`_,
    the source of the AbstractBackend class of which DaskBackend is a subclass.
    """

    # pylint: disable=no-self-use
    framework_name = "dask"

    def __init__(self):
        """Initialize DaskBackend.

        Contains the imports to avoid errors when dask is not installed
        """
        import dask.array as dsar

        self.dsar = dsar

    def is_appropriate_type(self, tensor):
        """Recognizes tensors it can handle."""
        return isinstance(tensor, self.dsar.core.Array)

    def from_numpy(self, x):  # noqa: D102
        return self.dsar.array(x)

    def to_numpy(self, x):  # noqa: D102
        return x.compute()

    def arange(self, start, stop):  # noqa: D102
        # supplementary method used only in testing, so should implement CPU version
        return self.dsar.arange(start, stop)

    def stack_on_zeroth_dimension(self, tensors: list):  # noqa: D102
        return self.dsar.stack(tensors)

    def tile(self, x, repeats):  # noqa: D102
        return self.dsar.tile(x, repeats)

    def is_float_type(self, x):  # noqa: D102
        return x.dtype in ("float16", "float32", "float64", "float128")

    def add_axis(self, x, new_position):  # noqa: D102
        return self.dsar.expand_dims(x, new_position)
