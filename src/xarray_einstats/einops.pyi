# File generated with docstub

import warnings
from collections.abc import Callable, Hashable

import einops
import xarray
import xarray as xr
from _typeshed import Incomplete

__all__ = ["rearrange", "reduce", "DaskBackend"]

class DimHandler:
    def __init__(self) -> None: ...
    def get_name(self, dim) -> None: ...
    def get_names(self, dim_list) -> None: ...
    def rename_kwarg(self, key) -> None: ...

def process_pattern_list(
    redims: list[Hashable | list | dict],
    handler: DimHandler,
    allow_dict: bool = ...,
    allow_list: bool = ...,
) -> tuple[list[str], list[str], str]: ...
def translate_pattern(pattern: str) -> list[Hashable | list | dict]: ...
def _rearrange(
    da: xarray.DataArray,
    out_dims: list[str, list | dict],
    in_dims: list[str | dict] | None = ...,
    dim_lengths: dict | None = ...,
) -> None: ...
def rearrange(
    da: xarray.DataArray,
    pattern: str | list[Hashable | list | dict],
    pattern_in: list[Hashable | dict] | None = ...,
    dim_lengths: dict[Hashable, int] | None = ...,
    **dim_lengths_kwargs: int,
) -> xarray.DataArray: ...
def _reduce(
    da: xarray.DataArray,
    reduction: str | Callable,
    out_dims: list[str | list | dict],
    in_dims: list[str | dict] | None = ...,
    dim_lengths: dict[Hashable, int] | None = ...,
) -> None: ...
def reduce(
    da: xarray.DataArray,
    pattern: str | list[str | list | dict],
    reduction: str | Callable,
    pattern_in: list[str | dict] | None = ...,
    dim_lengths: dict[Hashable, int] | None = ...,
    **dim_lengths_kwargs: int,
) -> xarray.DataArray: ...

class DaskBackend(einops._backends.AbstractBackend):

    framework_name: Incomplete

    def __init__(self) -> None: ...
    def is_appropriate_type(self, tensor) -> None: ...
    def from_numpy(self, x: Incomplete) -> None: ...
    def to_numpy(self, x: Incomplete) -> None: ...
    def arange(self, start: Incomplete, stop: Incomplete) -> None: ...
    def stack_on_zeroth_dimension(self, tensors: list) -> None: ...
    def tile(self, x: Incomplete, repeats: Incomplete) -> None: ...
    def is_float_type(self, x: Incomplete) -> None: ...
    def add_axis(self, x: Incomplete, new_position: Incomplete) -> None: ...
