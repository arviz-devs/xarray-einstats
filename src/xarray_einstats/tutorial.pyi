# File generated with docstub

from collections.abc import Sequence

import numpy as np
import xarray as xr
from _typeshed import Incomplete

try:
    from IPython import get_ipython
    from PIL.Image import fromarray

    array_display: Incomplete
except ModuleNotFoundError:
    array_display: Incomplete

def generate_mcmc_like_dataset(seed: int | Sequence[int] | None = ...) -> None: ...
def generate_matrices_dataarray(seed: int | Sequence[int] | None = ...) -> None: ...

if array_display:

    def display_np_arrays_as_images() -> None: ...
