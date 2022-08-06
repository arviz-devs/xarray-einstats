"""Utils for testing."""


def assert_dims_in_da(da, dims):
    """Assert all passed dims are present in the DataArray.

    If not true, the assertion error contains a list of all missing dims.
    """
    missing = [dim for dim in dims if dim not in da.dims]
    assert not missing, f"Some dims were missing: {missing}"


def assert_dims_not_in_da(da, dims):
    """Assert none of the passed dims are present in the DataArray.

    If not true, the assertion error contains a list of all incorrectly present dims.
    """
    present_but_shouldnt = [dim for dim in dims if dim in da.dims]
    assert not present_but_shouldnt, f"Found some of the provided dims: {present_but_shouldnt}"
