---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(getting_started)=
# Getting started

## Welcome to `xarray-einstats`!
`xarray-einstats` is an open source Python library part of the
{doc}`ArviZ project <arviz_org:index>`.
It acts as a bridge between the [xarray](https://xarray.dev/)
library for labelled arrays and libraries for raw arrays
such as [NumPy](https://numpy.org/) or [SciPy](https://scipy.org/).

Xarray has as "Compatibility with the broader ecosystem" as
one of its main {doc}`goals <xarray:getting-started-guide/why-xarray>`.
Which is what allows `xarray-einstats` to perform this
_bridge_ role with minimal code and duplication.

## Overview
`xarray-einstats` provides wrappers for:

* Most of the functions in {mod}`numpy.linalg`
* A subset of {mod}`scipy.stats`
* `rearrange` and `reduce` from [einops](http://einops.rocks/)

These wrappers have the same names and functionality as the original functions.
The difference in behaviour is that the wrappers will not make assumptions
about the meaning of a dimension based on its position
nor they have arguments like `axis` or `axes`.
They will have `dims` argument that take _dimension names_ instead of
integers indicating the positions of the dimensions on which to act.

It also provides a handful of re-implemented functions:

* {func}`xarray_einstats.numba.histogram`
* {class}`xarray_einstats.stats.multivariate_normal`

These are partially reimplemented because the original function
doesn't yet support multidimensional and/or batched computations.
They also share the name with a function in NumPy or SciPy,
but they only implement a subset of the features.
Moreover, the goal is for those to eventually be wrappers too.


## Using `xarray-einstats`
### DataArray inputs
Functions in `xarray-einstats` are designed to work on {class}`~xarray.DataArray` objects.

Let's load some example data:

```{code-cell} ipython3
from xarray_einstats import linalg, stats, tutorial

da = tutorial.generate_matrices_dataarray(4)
da
```

and show an example:

```{code-cell} ipython3
stats.skew(da, dims=["batch", "dim2"])
```

`xarray-einstats` uses `dims` as argument throughout the codebase
as an alternative to both `axis` or `axes` indistinctively,
also as alternative to the `(..., M, M)` convention used by NumPy.

The use of `dims` follows {func}`~xarray.dot`, instead of the singular
`dim` argument used for example in {meth}`~xarray.DataArray.mean`.
Both a single dimension or multiple are valid inputs,
and using `dims` emphasizes the fact that operations
and reductions can be performed over multiple dimensions at the same time.
Moreover, in linear algebra functions, `dims` is often restricted
to a 2 element list as it indicates which dimensions define the matrices,
interpreting all the others as batch dimensions.

That means that the two calls below are equivalent, even if the dimension
names of the inputs are not, _because their dimension names are the same_.
Thus,

```{code-cell} ipython3
linalg.det(da, dims=["dim", "dim2"])
```

returns the same as:

```{code-cell} ipython3
linalg.det(da.transpose("dim2", "experiment", "dim", "batch"), dims=["dim", "dim2"])
```

:::{important}
In `xarray_einstats` only the dimension names matter, not their order.
:::

### Dataset and GroupBy inputs
While the `DataArray` is the base xarray object, there are also
other xarray objects that are key while using the library.
These other objects such as {class}`~xarray.Dataset` are implemented as
a collection of `DataArray` objects, and all include a `.map`
method in order to apply the same function to all its child `DataArrays`.

```{code-cell} ipython3
ds = tutorial.generate_mcmc_like_dataset(9438)
ds
```

We can use {meth}`~xarray.Dataset.map` to apply the same function to
all the 4 child `DataArray`s in `ds`, but this will not always be possible.
When using `.map`, the function provided is applied to all child `DataArray`s
with the same `**kwargs`.

If we try doing:

```{code-cell} ipython3
:tags: [raises-exception, hide-output]

ds.map(stats.circmean, dims=("chain", "draw"))
```

we get an exception. The `chain` and `draw` dimensions are not present in all
child `DataArrays`. Instead, we could apply it only to the variables
that have both `chain` and `dim` dimensions.


```{code-cell} ipython3
ds_samples = ds[["mu", "sigma", "score"]]
ds_samples.map(stats.circmean, dims=("chain", "draw"))
```

:::{attention}
In general, you should prefer using `.map` attribute over using non-`DataArray` objects as
input to the `xarray_einstats` directly.
`.map` will ensure no unexpected broadcasting between the multiple child `DataArray`s takes place.
See the examples below for some examples.

However, if you are using functions that reduce dimensions on non-`DataArray` inputs
whose child `DataArray`s all have all the dimensions to reduce you will
not trigger any such broadcasting,
_and we have included that behaviour on our test suite to ensure it stays this way_.
:::

It is also possible to do


```{code-cell} ipython3
stats.circmean(ds_samples, dims=("chain", "draw"))
```

Here, all child `DataArray`s have both `chain` and `draw` dimension,
so as expected, the result is the same.
There are some cases however, in which _not_ using `.map` triggers
some broadcasting operations which will generally not be the desired
output.

If we use the `.map` attribute, the function is applied to each
child `DataArray` independently from the others:


```{code-cell} ipython3
ds.map(stats.rankdata)
```

whereas without using the `.map` attribute, extra broadcasting can happen:


```{code-cell} ipython3
stats.rankdata(ds)
```

---

The behaviour on {class}`~xarray.core.groupby.DataArrayGroupBy` for example is very similar
to the examples we have shown for `Dataset`s:


```{code-cell} ipython3
da = ds["mu"].assign_coords(team=["a", "b", "b", "a", "c", "b"])
da
```

when we apply a "group by" operation over the `team` dimension, we generate a
`DataArrayGroupBy` with 3 groups.

```{code-cell} ipython3
gb = da.groupby("team")
gb
```

on which we can use `.map` to apply a function from `xarray-einstats` over
all groups independently:

```{code-cell} ipython3
gb.map(stats.median_abs_deviation, dims=["draw", "team"])
```

which as expected has performed the operation group-wise, yielding a different
result than either

```{code-cell} ipython3
stats.median_abs_deviation(da, dims=["draw", "team"])
```

or

```{code-cell} ipython3
stats.median_abs_deviation(da, dims="draw")
```

:::{seealso}
Check out the {ref}`xarray:groupby` page on xarray's documentation.
:::
