# xarray-einstats

**Stats, linear algebra and einops for xarray**

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
### Key features

:::::{grid} 1 1 2 2
:gutter: 3

::::{grid-item-card}
Label aware
^^^

Apply operations over named dimensions.
Automatically aligns and broadcasts inputs,
and preserves dimensions and coordinates.
::::
::::{grid-item-card}
Interoperability
^^^

Wrappers in xarray-einstats are designed to be minimal
to preserve as many features from xarray as possible,
for example, Dask support.
::::
::::{grid-item-card}
Batched operations
^^^

All operations can be batched over one or multiple dimensions.
::::
::::{grid-item-card}
Flexible inputs
^^^

DataArrays, Datasets and even GroupBy xarray objects can
be used as inputs.
::::
:::::

::::{div} sd-text-center
:::{button-ref} getting_started
:color: primary
:outline:

Get started with xarray-einstats {material-round}`start;2em`
:::
::::

## Similar projects
Here we list some similar projects we know of. Note that all of
them are complementary and don't overlap:
* [xr-scipy](https://xr-scipy.readthedocs.io/en/latest/index.html)
* [xarray-extras](https://xarray-extras.readthedocs.io/en/latest/)
* [xhistogram](https://xhistogram.readthedocs.io/en/latest/)
* [xrft](https://xrft.readthedocs.io/en/latest/)

## Cite xarray-einstats
If you use this software, please cite it using the following template and the version
specific DOI provided by Zenodo. Click on the badge to go to the Zenodo page
and select the DOI corresponding to the version you used
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5895451.svg)](https://doi.org/10.5281/zenodo.5895451)

* Oriol Abril-Pla. (2022). arviz-devs/xarray-einstats `<version>`. Zenodo. `<version_doi>`

or in bibtex format:

```none
@software{xarray_einstats2022,
  author       = {Abril-Pla, Oriol},
  title        = {{xarray-einstats}},
  year         = 2022,
  url          = {https://github.com/arviz-devs/xarray-einstats}
  publisher    = {Zenodo},
  version      = {<version>},
  doi          = {<version_doi>},
}
```

:::{toctree}
:hidden:

installation
getting_started
tutorials/index
api/index
background/index
changelog
:::

```{toctree}
:caption: Contributing
:hidden:

Overview <contributing/overview>
Reference <contributing/dev_reference>
How-to guides <contributing/how_to>
```

```{toctree}
:caption: About
:hidden:

Twitter <https://twitter.com/arviz_devs>
Mastodon <https://bayes.club/@ArviZ>
GitHub repository <https://github.com/arviz-devs/xarray-einstats>
```
