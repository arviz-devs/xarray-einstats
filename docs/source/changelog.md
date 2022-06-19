# Change Log

## v.0.x.x (Unreleased)
### New features
* Add `DaskBackend` to support using einops functions on Dask backed DataArrays {pull}`14`

### Documentation
* Add dask support guide {pull}`14`
* Update requirements following [NEP 29](https://numpy.org/neps/nep-0029-deprecation_policy.html)
  and [SPEC 0](https://scientific-python.org/specs/spec-0000/) {pull}`19`

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
