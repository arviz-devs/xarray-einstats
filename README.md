# xarray-einstats

[![Documentation Status](https://readthedocs.org/projects/xarray-einstats/badge/?version=latest)](https://xarray-einstats.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
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
and intuitive. Here are some examples of such trade-off:

|  numpy  |  xarray  |
|---------|----------|
| `a[2, 5]` | `da.sel(drug="paracetamol", subject=5)` |
| `a.mean(axis=(0, 1))` | `da.mean(dim=("chain", "draw"))` |
| `` | `` |

In some other cases however, using xarray can result in overly verbose code
that often also becomes less clear. `xarray-einstats` provides wrappers
around some numpy and scipy functions (mostly `numpy.linalg` and `scipy.stats`)
and around [einops](https://einops.rocks/) with an api and features adapted to xarray.

% ⚠️  Attention: A nicer rendering of the content below is available at [our documentation](https://xarray-einstats.readthedocs.io/en/latest/)

### Data for examples
The examples in this overview page use the `DataArray`s from the `Dataset` below
(stored as `ds` variable) to illustrate `xarray-einstats` features:

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
`xarray-einstats` provides two wrapper classes {class}`xarray_einstats.XrContinuousRV`
and {class}`xarray_einstats.XrDiscreteRV` that can be used to wrap any distribution
in {mod}`scipy.stats` so they accept {class}`~xarray.DataArray` as inputs.

We can evaluate the logpdf using inputs that wouldn't align if using numpy
in a couple lines:

```python
norm_dist = xarray_einstats.XrContinuousRV(scipy.stats.norm)
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

### einops
**only rearrange wrapped for now**

[einops](https://einops.rocks/) uses a convenient notation inspired in
Einstein notation to specify operations on multidimensional arrays.
It uses spaces as a delimiter between dimensions, parenthesis to
indicate splitting or stacking of dimensions and `->` to separate
between input and output dim specification. `einstats` uses
an adapted notation then translates to einops and calls {func}`xarray.apply_ufunc`
under the hood.

Why change the notation? There are three main reasons, each concerning one
of the elements respectively: `->`, space as delimiter and parenthesis:
* In xarray dimensions are already labeled. In many cases, the left
  side in the einops notation is only used to label the dimensions.
  In fact, 5/7 examples in https://einops.rocks/api/rearrange/ fall in this category.
  This is not necessary when working with xarray objects.
* In xarray dimension names can be any {term}`xarray:hashable`. `xarray-einstats` only
  supports strings as dimension names, but the space can't be used as delimiter.
* In xarray dimensions are labeled and the order doesn't matter.
  This might seem the same as the first reason but it is not. When splitting
  or stacking dimensions you need (and want) the names of both parent and children dimensions.
  In some cases, for example stacking, we can autogenerate a default name, but
  in general you'll want to give a name to the new dimension. After all,
  dimension order in xarray doesn't matter and there isn't much to be done without knowing
  the dimension names.

`xarray-einstats` uses two separate arguments, one for the input pattern (optional) and
another for the output pattern. Each is a list of dimensions (strings)
or dimension operations (lists or dictionaries). Some examples:

We can combine the chain and draw dimensions and name the resulting dimension `sample`
using a list with a single dictionary. The `team` dimension is not present in the pattern
and is not modified.

```python
rearrange(ds.atts, [{"sample": ("chain", "draw")}])
```

Out:

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

```python
rearrange(ds.atts, [{"sample": ("chain", "draw")}, "team"])
```

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

Now to a more complicated pattern. We will split the chain and draw dimension,
then combine those split dimensions between them.

```python
rearrange(
    ds.atts,
    # combine split chain and team dims between them
    # here we don't use a dict so the new dimensions get a default name
    out_dims=[("chain1", "team1"), ("team2", "chain2")],
    # use dicts to specify which dimensions to split, here we *need* to use a dict
    in_dims=[{"chain": ("chain1", "chain2")}, {"team": ("team1", "team2")}],
    # set the lengths of split dimensions as kwargs
    chain1=2, chain2=2, team1=2, team2=3
)
```

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

More einops examples at {ref}`einops`

### Linear Algebra

**Still missing in the package**

There is no one size fits all solution, but knowing the function
we are wrapping we can easily make the code more concise and clear.
Without `xarray-einstats`, to invert a batch of matrices stored in a 4d
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

With `xarray-einstats`, those operations become:

```python
inv = xarray_einstats.inv(batch_of_matrices, dim=("matrix_dim", "matrix_dim_bis"))
norm = xarray_einstats.norm(batch_of_matrices, dim=("matrix_dim", "matrix_dim_bis"))
```



## Similar projects
Here we list some similar projects we know of. Note that all of
them are complementary and don't overlap:
* [xr-scipy](https://xr-scipy.readthedocs.io/en/latest/index.html)
* [xarray-extras](https://xarray-extras.readthedocs.io/en/latest/)
