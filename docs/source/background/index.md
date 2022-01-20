# Background

## Why xarray-einstats?
`xarray-einstats` aims to cover two important limitations one can encounter
when working with xarray objects:

### Labeled dimensions become a nuisance instead of being helpful

If you need to perform many specific or niche computations such as
linear algebra or statistical operations, you end up with a choice
between calling the functions as is or wrapping them with {func}`xarray.apply_ufunc`.
This means either losing (or ignoring) all the dims and coords information or
adding multiple lines of code _per operation_.

The half way solution between those two would be to group all operations
and wrap the result, which is actually the first case, we need to operate
on numpy arrays within the wrapped function so are forced to ignore
the labels. At the end of the day, we can't take advantage of the labels
during those generally critical computations yet we are still forced to
write longer code to not loose the info completely.

Moreover, the arguments to `apply_ufunc` are not intuitive
(nor they can be given the virtually endless flexibility they allow).
We end up writing longer more verbose code but it doesn't really become
more clear.

### Functions don't work on high dimensional data

Most functions were designed for 1-3D data and many have not been updated
to broadcast and handle multidimensional data properly. Usually this
is due to the functions requesting a specific number of dimensions
for the input arguments (i.e. the covariance matrix of a multidimensional
normal must be `(N, N)`, it can't be `(..., N, N)`) or because
the `axis`/`axes` argument only accepts integers.

When this happens, wrapping those functions and allowing users to call
them with labeled data and labels instead of axis numbers, they are
still not useful. If you have 4+ dimensions, chances are you'll need
to reduce two at a time at some point.

## Approach
We believe the best way to tackle those two issues if by wrapping
our target functions using `apply_ufunc` and extending them when necessary.
Extending can range from some stacking or automatic renaming
of dimensions to avoid conflicts and repeated names to using
`numba` to generate proper broadcasting ufuncs out of shape limited
functions.

Our goal is not to reimplement any of those functions ourselves,
and we will "move back" to a simpler extension once the upstream function
is improved/extended. Extensions however are still necessary because
upstreaming those changes take too much time and/or effort. In fact,
in many cases the process has already started, see this [GitHub issue](https://github.com/numpy/numpy/issues/17669)
about extending `numpy.random.multivariate_normal` for example,
but it is often hard to reach an agreement on what is the best way forward.

The general approach is to wrap our target functions using `apply_ufunc`,
which generally reduces to a 1-3 line wrapper changing `axis` for `dims`
that converts `dims` to `input/output_core_dims` as needed.
We believe even such a simple wrapper can have a huge impact as it makes
the code more clear and allows any user to use such functions, as many
might be intimidated by `apply_ufunc`.

The second approach is some minimal extension using xarray and string
methods. Some examples of this are
* stacking some dimensions of the input array,
  calling a function that accepts only integer `axis`,
  and then unstacking before returning the result
* creating or renaming new dimensions when dimensions are repeated
  or duplicated.

Lastly, the 3rd approach which is much more complex and requires significant
extra effort is to use `numba` or `jax` to extend the function to proper
support for multidimensional data, then wrap it using `apply_ufunc`.
As we have said, this is not our goal and we only resort to this option
for critical functions.

We also combine this layering approach to wrappers with strict modularity.
Wrappers are structured in completely independent modules
and not imported explicitly in the highest level namespace
(except for some numpy only wrappers).
This makes `xarray-einstats` depend only on xarray even if we
have modules wrapping functions from other libraries which are
then only optional dependencies.

More details about the design decisions and implementation details
of each module can be found in their specific "About ..." pages:

:::{toctree}
linalg
stats
einops
:::
