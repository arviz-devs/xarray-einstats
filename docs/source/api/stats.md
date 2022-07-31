# Stats
```{eval-rst}
.. automodule:: xarray_einstats.stats
```

## Probability distributions

:::{note}
These wrapper classes set some defaults and ensure
proper alignment and broadcasting of all inputs, but
use {func}`xarray.apply_ufunc` under the hood.
This means that while using kwargs for distribution
parameters is supported, using positional arguments
is recommended. In fact, if no positional arguments
are present, automatic broadcasting will still work
but the output will be a numpy array.
:::

```{eval-rst}
.. autosummary::
  :toctree: generated/

  XrContinuousRV
  XrDiscreteRV
  multivariate_normal
```

## Summary statistics
```{eval-rst}
.. autosummary::
  :toctree: generated/

  circmean
  circstd
  circvar
  gmean
  hmean
  kurtosis
  skew
  median_abs_deviation
```

## Other statistical functions

```{eval-rst}
.. autosummary::
  :toctree: generated/

  rankdata
```

## Convenience functions

```{eval-rst}
.. autosummary::
  :toctree: generated/

  get_default_dims
```
