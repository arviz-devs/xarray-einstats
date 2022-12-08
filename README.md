# xarray-einstats

[![Documentation Status](https://readthedocs.org/projects/xarray-einstats/badge/?version=latest)](https://xarray-einstats.readthedocs.io/en/latest/?badge=latest)
[![Run tests](https://github.com/arviz-devs/xarray-einstats/actions/workflows/test.yml/badge.svg)](https://github.com/arviz-devs/xarray-einstats/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/arviz-devs/xarray-einstats/branch/main/graph/badge.svg?token=78K2ZOJCVN)](https://codecov.io/gh/arviz-devs/xarray-einstats)
[![PyPI](https://img.shields.io/pypi/v/xarray-einstats)](https://pypi.org/project/xarray-einstats)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5895451.svg)](https://doi.org/10.5281/zenodo.5895451)


Stats, linear algebra and einops for xarray

## Installation

To install, run

```
(.venv) $ pip install xarray-einstats
```

See the docs for more [extensive install instructions](https://einstats.python.arviz.org/en/latest/installation.html).

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
Continue at the [getting started page](https://einstats.python.arviz.org/en/latest/getting_started.html).

## Contributing
xarray-einstats is in active development and all types of contributions are welcome!
See the [contributing guide](https://einstats.python.arviz.org/en/latest/contributing/overview.html) for details on how to contribute.

## Relevant links
* Documentation: https://einstats.python.arviz.org/en/latest/
* Contributing guide: https://einstats.python.arviz.org/en/latest/contributing/overview.html
* ArviZ project website: https://www.arviz.org

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
