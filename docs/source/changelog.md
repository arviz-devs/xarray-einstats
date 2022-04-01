# Change Log

## Unreleased (v0.2.0)
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

### Developer facing changes
* Added how-to release guide

## v0.1 (2022 Jan 24)
Initial version with modules: stats, linalg, einops and numba.
