# pylint: disable=redefined-outer-name, no-self-use
import numpy as np
import pytest
import xarray as xr

from xarray_einstats.einops import raw_rearrange, rearrange, translate_pattern

einops = pytest.importorskip("einops")  # pylint: disable=invalid-name


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(9)
    da = xr.DataArray(
        rng.normal(size=(4, 6, 15, 8)), dims=["batch", "subject", "experiment", "drug"]
    )
    return da


@pytest.mark.parametrize(
    "args",
    (
        ("subject drug home_team", ["subject", "drug", "home_team"]),
        ("( s1 s2 )", [["s1", "s2"]]),
        (" (s1 s2)=subject ", [{"subject": ["s1", "s2"]}]),
        ("( d1 d2 )=drug  (s1 s2)", [{"drug": ["d1", "d2"]}, ["s1", "s2"]]),
        (
            "a (s1 s2) b (d1 d2)=drug c",
            ["a", ["s1", "s2"], "b", {"drug": ["d1", "d2"]}, "c"],
        ),
    ),
)
def test_pattern_translation(args):
    pattern, result = args
    translation = translate_pattern(pattern)
    assert translation == result, (pattern, translation)


class TestRawRearrange:
    @pytest.mark.parametrize(
        "args",
        (
            ("(batch subject)=dim", {}, ((15, 8, 4 * 6), ["experiment", "drug", "dim"])),
            (
                "(e1 e2)=experiment -> e1 e2",
                dict(e1=3, e2=5),
                ((4, 6, 8, 3, 5), ["batch", "subject", "drug", "e1", "e2"]),
            ),
        ),
    )
    def test_raw_rearrange(self, data, args):
        pattern, kwargs, (shape, dims) = args
        out_da = raw_rearrange(data, pattern, **kwargs)
        assert out_da.shape == shape
        assert list(out_da.dims) == dims


class TestRearrange:
    @pytest.mark.parametrize(
        "args",
        (
            (
                dict(out_dims=[{"dex": ("drug dose (mg)", "experiment")}]),
                ((4, 6, 8 * 15), ["batch", "subject", "dex"]),
            ),
            (
                dict(
                    in_dims=[{"drug dose (mg)": ("d1", "d2")}],
                    out_dims=["d1", "d2", "batch"],
                    d1=2,
                    d2=4,
                ),
                ((6, 15, 2, 4, 4), ["subject", "experiment", "d1", "d2", "batch"]),
            ),
        ),
    )
    def test_rearrange(self, data, args):
        kwargs, (shape, dims) = args
        out_da = rearrange(data.rename({"drug": "drug dose (mg)"}), **kwargs)
        assert out_da.shape == shape
        assert list(out_da.dims) == dims
