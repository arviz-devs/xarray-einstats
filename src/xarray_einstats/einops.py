from collections.abc import Sequence

import xarray as xr
import einops

__all__ = ["rearrange"]


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
                raise ValueError(
                    "Found values of type str in a pattern dict, use xarray.rename"
                )
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


def rearrange(da, out_dims, in_dims=None, **kwargs):
    """Wrapper around einops.rearrange.

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
        In einops, the left side of the pattern serves two goals
    axes_lengths : dict, optional
        kwargs passed to einops.rearrange
    kwargs : dict, optional
        kwargs with key equal to dimension names in ``out_dims``
        (that is, strings or dict keys) are passed to einops.rearrange
        the rest of keys are passed to :func:`xarray.apply_ufunc`

    Notes
    -----
    Unlike for general xarray objects, where dimension
    names can be :term:`xarray:hashable` here
    dimension names are not recommended but required to be
    strings.

    See also
    --------
    xarray.DataArray.transpose
    xarray.Dataset.transpose
    """
    da_dims = da.dims

    handler = DimHandler()
    if in_dims is None:
        in_dims = []
        in_names = []
        in_pattern = ""
    else:
        in_dims, in_names, in_pattern = process_pattern_list(in_dims, handler=handler, allow_list=False)
    # note, not using sets for da_dims to avoid transpositions on missing variables,
    # if they wanted to transpose those they would not be missing variables
    out_dims, out_names, out_pattern = process_pattern_list(out_dims, handler=handler)
    missing_in_dims = [dim for dim in da_dims if dim not in in_names]
    expected_missing = set(out_dims).union(in_names).difference(in_dims)
    missing_out_dims = [dim for dim in da_dims if dim not in expected_missing]
    pattern = f"{handler.get_names(missing_in_dims)} {in_pattern} ->\
        {handler.get_names(missing_out_dims)} {out_pattern}"

    axes_lengths = {handler.rename_kwarg(k): v for k, v in kwargs.items() if k in out_names+out_dims}
    kwargs = {k: v for k, v in kwargs.items() if k not in out_names+out_dims}
    print(pattern)
    print((missing_in_dims, "+", in_names))
    print((missing_out_dims, "+", out_names))
    print(axes_lengths)
    print(kwargs)
    return xr.apply_ufunc(
        einops.rearrange,
        da,
        pattern,
        input_core_dims=[missing_in_dims + in_names, []],
        output_core_dims=[missing_out_dims + out_names],
        kwargs=axes_lengths,
        **kwargs,
    )
