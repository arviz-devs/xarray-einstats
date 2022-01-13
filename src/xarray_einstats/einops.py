"""Wrappers for `einops <https://einops.rocks/>`_."""
import einops
import xarray as xr

__all__ = ["rearrange", "raw_rearrange", "reduce", "raw_reduce"]


class DimHandler:
    def __init__(self):
        self.mapping = {}

    def get_name(self, dim):
        if dim in self.mapping:
            return self.mapping.get(dim)
        dim_txt = f"d{len(self.mapping)}"
        self.mapping[dim] = dim_txt
        return dim_txt

    def get_names(self, dim_list):
        return " ".join((self.get_name(dim) for dim in dim_list))

    def rename_kwarg(self, key):
        return self.mapping.get(key, key)


def process_pattern_list(redims, handler, allow_dict=True, allow_list=True):
    out = []
    out_names = []
    txt = []
    for subitem in redims:
        if isinstance(subitem, str):
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
            if isinstance(values, str):
                raise ValueError("Found values of type str in a pattern dict, use xarray.rename")
            out.extend(values)
            out_names.append(key)
            txt.append(f"( {handler.get_names(values)} )")
        elif allow_list:
            out.extend(subitem)
            out_names.append(",".join(subitem))
            txt.append(f"( {handler.get_names(subitem)} )")
        else:
            raise ValueError(
                f"Found unsupported pattern type: {type(subitem)}, double check the docs. "
                "This could be for example is using lists/tuples as elements of in_dims argument"
            )
    return out, out_names, " ".join(txt)


def translate_pattern(pattern):
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


def rearrange(da, out_dims, in_dims=None, **kwargs):
    """Wrap `einops.rearrange <https://einops.rocks/api/rearrange/>`_.

    Parameters
    ----------
    da : xarray.DataArray
        Input DataArray to be rearranged
    out_dims : list of str, list or dict
        The output pattern for the dimensions.
        The dimensions present in
    in_dims : list of str or dict, optional
        The input pattern for the dimensions.
        This is only necessary if you want to split some dimensions.
    kwargs : dict, optional
        kwargs with key equal to dimension names in ``out_dims``
        (that is, strings or dict keys) are passed to einops.rearrange
        the rest of keys are passed to :func:`xarray.apply_ufunc`

    Notes
    -----
    Unlike for general xarray objects, where dimension
    names can be :term:`hashable <xarray:name>` here
    dimension names are not recommended but required to be
    strings.

    See Also
    --------
    xarray_einstats.einops.raw_rearrange:
        Cruder wrapper of einops.rearrange, allowed characters in dimension names are restricted
    xarray.DataArray.transpose, xarray.Dataset.transpose
    xarray.DataArray.stack, xarray.Dataset.stack
    """
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
    pattern = f"{handler.get_names(missing_in_dims)} {in_pattern} ->\
        {handler.get_names(missing_out_dims)} {out_pattern}"

    axes_lengths = {
        handler.rename_kwarg(k): v for k, v in kwargs.items() if k in out_names + out_dims
    }
    kwargs = {k: v for k, v in kwargs.items() if k not in out_names + out_dims}
    return xr.apply_ufunc(
        einops.rearrange,
        da,
        pattern,
        input_core_dims=[missing_in_dims + in_names, []],
        output_core_dims=[missing_out_dims + out_names],
        kwargs=axes_lengths,
        **kwargs,
    )


def raw_rearrange(da, pattern, **kwargs):
    """Crudely wrap `einops.rearrange <https://einops.rocks/api/rearrange/>`_.

    Wrapper around einops.rearrange with a very similar syntax.
    Spaces, parenthesis ``()`` and `->` are not allowed in dimension names.

    Parameters
    ----------
    da : xarray.DataArray
        Input array
    pattern : string
        Pattern string. Same syntax as patterns in einops with two
        caveats:

        * Unless splitting or stacking, you must use the actual dimension names.
        * When splitting or stacking you can use `(dim1 dim2)=dim`. This is `necessary`
          for the left hand side as it identifies the dimension to split, and
          optional on the right hand side, if omitted the stacked dimension will be given
          a default name.

    kwargs : dict, optional
        Passed to :func:`xarray_einstats.einops.rearrange`

    Returns
    -------
    xarray.DataArray

    See Also
    --------
    xarray_einstats.einops.rearrange:
        More flexible and powerful wrapper over einops.rearrange. It is also more verbose.
    """
    if "->" in pattern:
        in_pattern, out_pattern = pattern.split("->")
        in_dims = translate_pattern(in_pattern)
    else:
        out_pattern = pattern
        in_dims = None
    out_dims = translate_pattern(out_pattern)
    return rearrange(da, out_dims=out_dims, in_dims=in_dims, **kwargs)


def reduce(da, reduction, out_dims, in_dims=None, **kwargs):
    """Wrap `einops.reduce <https://einops.rocks/api/reduce/>`_.

    Parameters
    ----------
    da : xarray.DataArray
        Input DataArray to be reduced
    reduction : string or callable
        One of available reductions ('min', 'max', 'sum', 'mean', 'prod') by ``einops.reduce``,
        case-sensitive. Alternatively, a callable ``f(tensor, reduced_axes) -> tensor``
        can be provided. ``reduced_axes`` are passed as a list of int.
    out_dims : list of str, list or dict
        The output pattern for the dimensions.
        The dimensions present in
    in_dims : list of str or dict, optional
        The input pattern for the dimensions.
        This is only necessary if you want to split some dimensions.
    kwargs : dict, optional
        kwargs with key equal to dimension names in ``out_dims``
        (that is, strings or dict keys) are passed to einops.rearrange
        the rest of keys are passed to :func:`xarray.apply_ufunc`

    Notes
    -----
    Unlike for general xarray objects, where dimension
    names can be :term:`hashable <xarray:name>` here
    dimension names are not recommended but required to be
    strings.

    See Also
    --------
    xarray_einstats.einops.raw_reduce:
        Cruder wrapper of einops.rearrange, allowed characters in dimension names are restricted
    xarray_einstats.einops.rearrange, xarray_einstats.einops.raw_rearrange
    """
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
    axes_lengths = {handler.rename_kwarg(k): v for k, v in kwargs.items() if k in all_dims}
    kwargs = {k: v for k, v in kwargs.items() if k not in all_dims}
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


def raw_reduce(da, pattern, reduction, **kwargs):
    """Crudely wrap `einops.reduce <https://einops.rocks/api/reduce/>`_.

    Wrapper around einops.reduce with a very similar syntax.
    Spaces, parenthesis ``()`` and `->` are not allowed in dimension names.

    Parameters
    ----------
    da : xarray.DataArray
        Input array
    pattern : string
        Pattern string. Same syntax as patterns in einops with two
        caveats:

        * Unless splitting or stacking, you must use the actual dimension names.
        * When splitting or stacking you can use `(dim1 dim2)=dim`. This is `necessary`
          for the left hand side as it identifies the dimension to split, and
          optional on the right hand side, if omitted the stacked dimension will be given
          a default name.

    reduction : string or callable
        One of available reductions ('min', 'max', 'sum', 'mean', 'prod') by ``einops.reduce``,
        case-sensitive. Alternatively, a callable ``f(tensor, reduced_axes) -> tensor``
        can be provided. ``reduced_axes`` are passed as a list of int.
    kwargs : dict, optional
        Passed to :func:`xarray_einstats.einops.reduce`

    Returns
    -------
    xarray.DataArray

    See Also
    --------
    xarray_einstats.einops.reduce:
        More flexible and powerful wrapper over einops.reduce. It is also more verbose.
    xarray_einstats.einops.rename_kwarg, xarray_einstats.einops.raw_rearrange
    """
    if "->" in pattern:
        in_pattern, out_pattern = pattern.split("->")
        in_dims = translate_pattern(in_pattern)
    else:
        out_pattern = pattern
        in_dims = None
    out_dims = translate_pattern(out_pattern)
    return reduce(da, reduction, out_dims=out_dims, in_dims=in_dims, **kwargs)
