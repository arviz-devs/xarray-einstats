(about_stats)=
# About the stats module
The stats module is composed of the wrappers for statistical distributions
and wrappers for other statistical function and summaries. These two
follow completely different approaches which are discussed in
the two sections below, one per block.

This page assumes you have already read {ref}`stats_tutorial`.

## About distribution wrappers
There are only two classes that serve as wrappers to _all_ statistical
distributions in {mod}`scipy.stats`, one for continuous distributions
and another for discrete ones. This has some drawbacks related to preservation
of named dimensions and coordinates, but we believe its simplicity and ease of
maintenance outweighs this drawbacks.

The two wrappers {class}`~xarray_einstats.stats.XrContinuousRV` and
{class}`~xarray_einstats.stats.XrDiscreteRV` take as first
argument the scipy distribution to be wrapped and then optional
args and kwargs. Methods take also a mixture of args and kwargs,
mimicking the behaviour of scipy distributions where the
scale for example can be defined either at creation time
or when calling a method and it can be passed as a positional
or keyword argument.

The xarray einstats wrapper classes however, instead of initializing the distributions
at creation store the distribution and initialization args and kwargs. Then,
whenever a method is called the args and kwargs provided via the method
and the ones provided at initialization are combined and broadcasted.
This happens in the `_broadcast_args` method of {class}`xarray_einstats.stats.XrRV`.
The combined+broadcasted arguments are used to call the scipy distribution via
{func}`xarray.apply_ufunc`, which ensures that the shapes will be compatible.

As the same wrappers are used for all distributions, even if both positional and
keyword arguments are broadcasted, they are used as provided when calling `apply_ufunc`.
The main drawback of this approach is that `apply_ufunc` is only able to preserve
the dimensions and coordinates of _positional_ arguments. Therefore, given
two equivalent wrappers, one using positional and another using keyword arguments,
there are some edge cases where the one using keyword arguments will return numpy
arrays instead of `DataArray`s. Values are the same in both cases, but one
case has lost all information about named dimensions and coordinates.
The arguably more common and annoying case of such behaviour is with the `.rvs` method.

```{jupyter-execute}
import numpy as np
from scipy import stats
from xarray_einstats.tutorial import generate_mcmc_like_dataset
from xarray_einstats import stats as xtats
ds = generate_mcmc_like_dataset(3)

dist_pos = xtats.XrContinuousRV(stats.norm, ds["mu"], ds["sigma"])
dist_kw = xtats.XrContinuousRV(stats.norm, loc=ds["mu"], scale=ds["sigma"])

rvs_pos = dist_pos.rvs(size=5, random_state=7)
rvs_kw = dist_kw.rvs(size=5, random_state=7)
allclose = np.allclose(rvs_pos, rvs_kw)

print(f"Output type of rv_pos: {type(rvs_pos)}")
print(f"Output type of rv_kw:  {type(rvs_kw)}")
print(f"\nCheck all values are indeed equal in both cases: {allclose}")
```

In other methods, this is more complicated to trigger, because only one positional
argument is enough to preserve _all_ information. As the rest of the methods
convert array input to xarray under the hood, the following code doesn't lose any
labels:

```{jupyter-execute}
dist_kw.pdf(np.linspace(-3, 3))
```

## About statistical function and summary wrappers
Most wrappers here are minimal wrappers, that generally spend more time handling argument defaults.
The general pattern of these wrappers is the following:
1. Handle arguments. Arguments whose information is not needed by the wrapper generally default
  to `None` and are not included in the `dict` passed to `apply_ufunc` as `kwargs` argument.
  This covers us from having to track changes in scipy and update our argument defaults.
2. (optional) Take care of arguments that accept array values, that in `xarray_einstats`
  take `DataArray`s. They are broadcasted and aligned so the computation works.
3. Stack/reshape _if necessary_. `xarray_einstats` uses a `dims` argument that differs from
  scipy `axis` because it takes strings and because sequences of strings are also valid. When
  multiple dimensions are provided via `dims` they are stacked before calling scipy as
  it only takes integer `axis`
4. Call the scipy function.

Steps 3 and 4 are generally done via {func}`xarray_einstats.stats._apply_reduce_func` and
{func}`xarray_einstats.stats._apply_nonreduce_func`. However, if necessary, they are
done manually (for now this only happens with {func}`~xarray_einstats.stats.median_abs_deviance`)
