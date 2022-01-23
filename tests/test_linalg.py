import numpy as np
import pytest
import xarray as xr

from xarray_einstats import tutorial


@pytest.fixture(scope="module")
def data():
    ds = xr.Dataset(
        {
            "a": (("chain", "draw", "team"), np.full((4, 5, 3), 2)),
            "b": (("chain", "draw"), np.ones((4, 5))),
        }
    )
    return ds


@pytest.fixture(scope="module")
def matrices():
    ds = tutorial.generate_matrices_dataarray(3)
    return ds
