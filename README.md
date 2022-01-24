# xarray-einstats

[![Documentation Status](https://readthedocs.org/projects/xarray-einstats/badge/?version=latest)](https://xarray-einstats.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/arviz-devs/xarray-einstats/branch/main/graph/badge.svg?token=78K2ZOJCVN)](https://codecov.io/gh/arviz-devs/xarray-einstats)
[![PyPI](https://img.shields.io/pypi/v/xarray-einstats)](https://pypi.org/project/xarray-einstats)

Stats, linear algebra and einops for xarray

> ⚠️  **Caution: This project is still in a very early development stage**

## Installation

To install, run

```
(.venv) $ pip install xarray-einstats
```

## Overview
As stated in their website:

> xarray makes working with multi-dimensional labeled arrays simple, efficient and fun!

The code is often more verbose, but it is generally because it is clearer and thus less error prone
and more intuitive.
Here are some examples of such trade-off where we believe the increased clarity is worth
the extra characters:


|  numpy  |  xarray  |
|---------|----------|
| `a[2, 5]` | `da.sel(drug="paracetamol", subject=5)` |
| `a.mean(axis=(0, 1))` | `da.mean(dim=("chain", "draw"))` |
| `a.reshape((-1, 10))`  | `da.stack(sample=("chain", "draw"))` |
| `a.transpose(2, 0, 1)` | `da.transpose("drug", "chain", "draw")` |

In some other cases however, using xarray can result in overly verbose code
that often also becomes less clear. `xarray_einstats` provides wrappers
around some numpy and scipy functions (mostly `numpy.linalg` and `scipy.stats`)
and around [einops](https://einops.rocks/) with an api and features adapted to xarray.

% ⚠️  Attention: A nicer rendering of the content below is available at [our documentation](https://xarray-einstats.readthedocs.io/en/latest/)

### Data for examples
The examples in this overview page use the `DataArray`s from the `Dataset` below
(stored as `ds` variable) to illustrate `xarray_einstats` features:

```none
<xarray.Dataset>
Dimensions:  (dim_plot: 50, chain: 4, draw: 500, team: 6)
Coordinates:
  * chain    (chain) int64 0 1 2 3
  * draw     (draw) int64 0 1 2 3 4 5 6 7 8 ... 492 493 494 495 496 497 498 499
  * team     (team) object 'Wales' 'France' 'Ireland' ... 'Italy' 'England'
Dimensions without coordinates: dim_plot
Data variables:
    x_plot   (dim_plot) float64 0.0 0.2041 0.4082 0.6122 ... 9.592 9.796 10.0
    atts     (chain, draw, team) float64 0.1063 -0.01913 ... -0.2911 0.2029
    sd_att   (draw) float64 0.272 0.2685 0.2593 0.2612 ... 0.4112 0.2117 0.3401
```

### Stats
{mod}`xarray_einstats.stats` provides two wrapper classes {class}`xarray_einstats.stats.XrContinuousRV`
and {class}`xarray_einstats.stats.XrDiscreteRV` that can be used to wrap any distribution
in {mod}`scipy.stats` so they accept {class}`~xarray.DataArray` as inputs,
and some wrappers for other functions in the `scipy.stats` module
so you can use `dims` (supporting both string and iterable of strings)
instead of `axis` and keep the labels from the input DataArrays.

The distribution wrappers perform broadcasting and alignment of
all the inputs automatically.
You can evaluate the logpdf using inputs that wouldn't align if using numpy
in a couple lines:

```python
norm_dist = xarray_einstats.stats.XrContinuousRV(scipy.stats.norm)
#   shapes:         (50,)     (4, 500, 6)     (500,)
norm_dist.logpdf(ds["x_plot"], ds["atts"], ds["sd_att"])
```

which returns:

```none
<xarray.DataArray (dim_plot: 50, chain: 4, draw: 500, team: 6)>
array([[[[ 3.06470249e-01,  3.80373065e-01,  2.56575936e-01,
...
          -4.41658154e+02, -4.57599982e+02, -4.14709280e+02]]]])
Coordinates:
  * chain    (chain) int64 0 1 2 3
  * draw     (draw) int64 0 1 2 3 4 5 6 7 8 ... 492 493 494 495 496 497 498 499
  * team     (team) object 'Wales' 'France' 'Ireland' ... 'Italy' 'England'
Dimensions without coordinates: dim_plot
```

More examples available at {ref}`stats_tutorial`.

### Linear Algebra

There is no one size fits all solution, but knowing the function
we are wrapping we can easily make the code more concise and clear.
Without `xarray_einstats`, to invert a batch of matrices stored in a 4d
array you have to do:

```python
inv = xarray.apply_ufunc(   # output is a 4d labeled array
    numpy.linalg.inv,
    batch_of_matrices,      # input is a 4d labeled array
    input_core_dims=[["matrix_dim", "matrix_dim_bis"]],
    output_core_dims=[["matrix_dim", "matrix_dim_bis"]]
)
```

to calculate it's norm instead, it becomes:

```python
norm = xarray.apply_ufunc(  # output is a 2d labeled array
    numpy.linalg.norm,
    batch_of_matrices,      # input is a 4d labeled array
    input_core_dims=[["matrix_dim", "matrix_dim_bis"]],
)
```

With {mod}`xarray_einstats.linalg`, those operations become:

```python
inv = xarray_einstats.inv(batch_of_matrices, dim=("matrix_dim", "matrix_dim_bis"))
norm = xarray_einstats.norm(batch_of_matrices, dim=("matrix_dim", "matrix_dim_bis"))
```

Moreover, if you use some internal conventions to label the dimensions
that correspond to matrices, so that they can always be identified
if given the list of all dimensions in the input, you can configure
`xarray_einstats` to follow that convention.
Take a look at {func}`~xarray_einstats.linalg.get_default_dims`

And if you still need more reasons for `xarray_einstats`, to complement
the `einops` wrappers, it also provides {func}`xarray_einstats.einsum`!

More examples available, also using `einsum` at {ref}`linalg_tutorial`.

### einops
**repeat wrapper still missing**

[einops](https://einops.rocks/) uses a convenient notation inspired in
Einstein notation to specify operations on multidimensional arrays.
It uses spaces as a delimiter between dimensions, parenthesis to
indicate splitting or stacking of dimensions and `->` to separate
between input and output dim specification.

{mod}`xarray_einstats.einops` uses an adapted notation to take advantage of xarray,
where dimensions are already labeled,
and adapts to dimension names with spaces or parenthesis in them.
It then translates the expression and calls einops via {func}`xarray.apply_ufunc`
so you need to have einops installed for the functions in this
module to work.

`xarray_einstats` uses two separate arguments, one for the input pattern (optional) and
another for the output pattern. Each is a list of dimensions (strings)
or dimension  (lists or dictionaries).

:::{tip}
If you are willing to impose some extra constraints to your dimension names,
you can also use the `raw_` einops wrappers, with a syntax more concise and
much closer to the einops library.
:::

**Combine the chain and draw dimensions**

::::{tab-set}
:::{tab-item} rearrange
:sync: full

We can combine the chain and draw dimensions and name the resulting dimension `sample`
using a list with a single dictionary.

```python
rearrange(ds.atts, [{"sample": ("chain", "draw")}])
```
:::
:::{tab-item} raw_rearrange
:sync: raw

As you would do in einops, we indicate we want to combine the chain and draw dimensions
by putting the two inside a parenthesis. With `xarray_einstats` in addition,
you can add an `=new_name` to label this combined dimension, otherwise it gets
a default name.

Moreover, as dimensions are already labeled in the input, we can skip the
left side of the expression. If no `->` symbol is present in the pattern,
`xarray_einstats` generates the left side automatically.

```python
raw_rearrange(ds.atts, "(chain draw)=sample")
```
:::
::::

The `team` dimension is not present in the pattern and is not modified.
As here dimensions are named already in the input object, we don't need
ellipsis nor adding dimensions in both input and output to indicate they
are left as is. You can see how the team dimension has not been modified
in the output below:

```none
<xarray.DataArray 'atts' (team: 6, sample: 2000)>
array([[ 0.10632395,  0.1538294 ,  0.17806237, ...,  0.16744257,
         0.14927569,  0.21803568],
         ...,
       [ 0.30447644,  0.22650416,  0.25523419, ...,  0.28405435,
         0.29232681,  0.20286656]])
Coordinates:
  * team     (team) object 'Wales' 'France' 'Ireland' ... 'Italy' 'England'
Dimensions without coordinates: sample
```

Note that following xarray convention, new dimensions and dimensions on which we operated
are moved to the end. This only matters when you access the underlying array with `.values`
or `.data` and you can always transpose using {meth}`xarray.Dataset.transpose`, but
it can matter. You can change the pattern to enforce the output dimension order:

::::{tab-set}
:::{tab-item} rearrange
:sync: full
```python
rearrange(ds.atts, [{"sample": ("chain", "draw")}, "team"])
```
:::
:::{tab-item} raw_rearrange
:sync: raw
```python
raw_rearrange(ds.atts, "(chain draw)=sample team")
```
:::
::::

Out:

```none
<xarray.DataArray 'atts' (sample: 2000, team: 6)>
array([[ 0.10632395, -0.01912607,  0.13671159, -0.06754783, -0.46083807,
         0.30447644],
       ...,
       [ 0.21803568, -0.11394285,  0.09447937, -0.11032643, -0.29111234,
         0.20286656]])
Coordinates:
  * team     (team) object 'Wales' 'France' 'Ireland' ... 'Italy' 'England'
Dimensions without coordinates: sample
```

**Decompose and combine two dimensions in a different order**

Now to a more complicated pattern. We will split the chain and team dimension,
then combine those split dimensions between them.

::::{tab-set}
:::{tab-item} rearrange
:sync: full

Use a list of dictionaries to choose which dimensions to decompose,
note that lists with dimensions to decompose are not valid, you
_need_ to indicate which dimension is the one to be decomposed.

```python
rearrange(
    ds.atts,
    in_dims=[{"chain": ("chain1", "chain2")}, {"team": ("team1", "team2")}],
    # combine split chain and team dims between them
    # here we don't use a dict so the new dimensions get a default name
    out_dims=[("chain1", "team1"), ("team2", "chain2")],
    # set the lengths of split dimensions as kwargs
    chain1=2, chain2=2, team1=2, team2=3
)
```
:::
:::{tab-item} raw_rearrange
:sync: raw

We use `()=` on the left side because we _need_ to indicate which dimensions
to decompose, but we can skip it if we want on the right side and `xarray_einstats`
uses a default name for them.
```python
raw_rearrange(
    ds.atts,
    "(chain1 chain2)=chain (team1 team2)=team -> (chain1 team1) (team2 chain2)",
    # set the lengths of split dimensions as kwargs
    chain1=2, chain2=2, team1=2, team2=3
)
```
:::
::::

Out:

```none
<xarray.DataArray 'atts' (draw: 500, chain1,team1: 4, team2,chain2: 6)>
array([[[ 1.06323952e-01,  2.47005252e-01, -1.91260714e-02,
         -2.55769582e-02,  1.36711590e-01,  1.23165119e-01],
...
        [-2.76616968e-02, -1.10326428e-01, -3.99582340e-01,
         -2.91112341e-01,  1.90714405e-01,  2.02866563e-01]]])
Coordinates:
  * draw     (draw) int64 0 1 2 3 4 5 6 7 8 ... 492 493 494 495 496 497 498 499
Dimensions without coordinates: chain1,team1, team2,chain2
```

More einops examples with both `rearrange` and `reduce` at {ref}`einops_tutorial`

### Other features
`xarray_einstats` also includes some functions that are not direct wrappers of other
libraries. {func}`~xarray_einstats.numba.histogram` for example combines numba,
numpy and xarray to provide a vectorized version of `numpy.histogram` that works
on DataArrays.

## Similar projects
Here we list some similar projects we know of. Note that all of
them are complementary and don't overlap:
* [xr-scipy](https://xr-scipy.readthedocs.io/en/latest/index.html)
* [xarray-extras](https://xarray-extras.readthedocs.io/en/latest/)
