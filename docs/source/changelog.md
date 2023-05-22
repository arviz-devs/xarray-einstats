# Change Log

## v0.x.x (Unreleased)
### New features

### Maintenance and fixes
* Update dependencies and follow new pylint recommendations {pull}`49`

### Documentation

## v0.5.1 (2023 Jan 20)
### Maintenance and fixes
* Fix lower cap on xarray version dependency {pull}`45`

## v0.5.0 (2023 Jan 16)
### New features
* Added {func}`.empty_ref`, {func}`.ones_ref` and {func}`.zeros_ref` DataArray creation helpers {pull}`37`
* Added {func}`.linalg.diagonal` wrapper {pull}`37`
* Added {func}`.stats.logsumexp` wrapper {pull}`40`
* Added {func}`.searchsorted` and {func}`.ecdf` in {mod}`~xarray_einstats.numba` module {pull}`40`
* Added {func}`~xarray_einstats.sort` wrapper for vectorized sort along specific dimension using values {pull}`40`

### Maintenance and fixes
* Fix issue in `linalg.svd` for non-square matrices {pull}`37`
* Fix evaluation of distribution methods (e.g. `.pdf`) on scalars {pull}`38` and {pull}`39`
* Ensure support on inputs with stacked dimensions {pull}`40`

### Documentation
* Ported NumPy tutorial on linear algebra with multidimensional arrays {pull}`37`
* Added ecdf usage example and plotting reference {pull}`40`

## v0.4.0 (2022 Dec 9)
### New features
* Add `multivariate_normal` distribution class {pull}`23`

### Maintenance and fixes
* Update pyproject.toml to support building the package with both `flit` and `setuptools` {pull}`26`
* Improve testing structure and configuration {pull}`28`
* Update python and dependency versions {pull}`33`

### Documentation
* Add a section on running tests locally in contributing docs {pull}`28`
* Add a getting started page to the docs {pull}`30`
* Improve installation page {pull}`33`
* Separate and improve README and index pages {pull}`33`

## v.0.3.0 (2022 Jun 19)
### New features
* Add `DaskBackend` to support using einops functions on Dask backed DataArrays {pull}`14`

### Maintenance and fixes
* Update requirements following [NEP 29](https://numpy.org/neps/nep-0029-deprecation_policy.html)
  and [SPEC 0](https://scientific-python.org/specs/spec-0000/) {pull}`19`

### Documentation
* Add Dask support guide {pull}`14`
* Add references to xhistogram and xrft in docs {pull}`20`

## v0.2.2 (2022 Apr 3)
### Maintenance and fixes
* Add license file to `pyproject.toml` and remove ignored manifest file {pull}`13`

## v0.2.1 (2022 Apr 3)
### Maintenance and fixes
* Add manifest file to include the license and changelog in the pypi package {pull}`12`

## v0.2.0 (2022 Apr 2)
### New Features
* Added `skew`, `kurtosis` and `median_abs_deviation` to `stats` module {pull}`3`, {pull}`4`
* Improve flexibility and autorenaming in `matmul` {pull}`8`

### Maintenance and fixes
* Changed the automatic names in einops module to use dashes instead of commas
* Make API coherent with the call signature `da, dims=None, *, ...` for `stats`
  and `linalg` modules {pull}`7`.
* Add tests on using summary stats on `xarray.Dataset` objects {pull}`9`

### Documentation
* Added info on how to cite the library on README and citation file
* Added background/explanation page for the stats module. {pull}`5`
* Improve citation guidance {pull}`6`
* Add examples of summary stats usage on `xarray.Dataset` {pull}`9`
* Update the installation instructions with stable and development commands {pull}`10`

### Developer facing changes
* Added how-to release guide

## v0.1 (2022 Jan 24)
Initial version with modules: stats, linalg, einops and numba.
