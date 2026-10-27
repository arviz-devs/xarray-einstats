"""Check linear solves against their named-dimension equations."""

import pytest
import xarray as xr
from xarray.testing import assert_allclose

from xarray_einstats.linalg import solve


@pytest.mark.parametrize("shared_dim", ("row", "col"))
@pytest.mark.parametrize("multiple_rhs", (False, True))
@pytest.mark.parametrize("batched", (False, True))
def test_solve_shared_dimension(shared_dim, multiple_rhs, batched):
    """Either matrix dimension can be shared with the right-hand side."""
    matrix = xr.DataArray(
        [[3.0, 1.0], [2.0, 4.0]],
        dims=("row", "col"),
        coords={"row": ["r0", "r1"], "col": ["c0", "c1"]},
    )
    solution_dim = "col" if shared_dim == "row" else "row"
    expected = xr.DataArray(
        [1.0, 2.0], dims=solution_dim, coords={solution_dim: matrix[solution_dim]}
    )
    if multiple_rhs:
        columns = xr.DataArray([1.0, 2.0, 3.0], dims="rhs", coords={"rhs": ["a", "b", "c"]})
        expected = expected * columns
    if batched:
        scale = xr.DataArray([1.0, 2.0], dims="batch", coords={"batch": ["first", "second"]})
        matrix = (matrix * scale).transpose("col", "batch", "row")
        expected = scale * expected
    rhs = xr.dot(matrix, expected, dim=solution_dim)
    # Physical axis order must not determine the equation being solved.
    rhs = rhs.transpose(*reversed(rhs.dims))

    result = solve(matrix, rhs, dims=("row", "col", "rhs") if multiple_rhs else ("row", "col"))

    assert_allclose(result, expected)
    reconstructed = xr.dot(matrix, result, dim=solution_dim)
    assert_allclose(reconstructed.transpose(*rhs.dims), rhs)
    reversed_result = solve(
        matrix, rhs, dims=("col", "row", "rhs") if multiple_rhs else ("col", "row")
    )
    assert_allclose(reversed_result, expected)
    accessor_result = matrix.linalg.solve(
        rhs, dims=("row", "col", "rhs") if multiple_rhs else ("row", "col")
    )
    assert_allclose(accessor_result, expected)
