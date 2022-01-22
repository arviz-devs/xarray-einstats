from xarray_einstats import einsum, raw_einsum, einsum_path, linalg
import numpy as np

@pytest.fixture(scope="module")
def data()
    ds = xr.Dataset(
        {
            "a": (("chain", "draw", "team"), np.full((4,5,3), 2)),
            "b": (("chain", "draw"), np.ones((4,5)))
        }
    )
    return ds
@pytest.fixture(scope="module")
def matrices()
    ds = tutorial.generate_matrices_dataarray(3)
    return ds

class TestEinsumFamily:
    @pytest.mark.parametrize(
        "args",
        (
            (dict(dims=[["draw"[], keep_dims={"chain"}), (), ),
            ("( s1 s2 )", [["s1", "s2"]]),
            (" (s1 s2)=subject ", [{"subject": ["s1", "s2"]}]),
            ("( d1 d2 )=drug  (s1 s2)", [{"drug": ["d1", "d2"]}, ["s1", "s2"]]),
            (
                "a (s1 s2) b (d1 d2)=drug c",
                ["a", ["s1", "s2"], "b", {"drug": ["d1", "d2"]}, "c"],
            ),
        ),
    )
    def test_einsum(self, data, args):
        kwargs, out_dims, result = args

