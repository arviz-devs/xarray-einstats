# Linear Algebra

```{eval-rst}
.. automodule:: xarray_einstats.linalg
```

:::{tip}
Most of the functions in this module are also available via the `.linalg` accessor
from :class:`DataArray` objects. See {ref}`linalg_tutorial` for
example usage.

The functions that are not available via the accessor are `einsum`, `einsum_path`,
`matmul` and `get_default_dims`.
:::

## Matrix and vector products
```{eval-rst}
.. currentmodule:: xarray_einstats
.. autosummary::
  :toctree: generated/

  einsum
  einsum_path
  matmul
  linalg.matrix_transpose
  linalg.matrix_power
```

## Decompositions
```{eval-rst}
.. currentmodule:: xarray_einstats.linalg
.. autosummary::
  :toctree: generated/

  cholesky
  qr
  svd
```

## Matrix eigenvalues
```{eval-rst}
.. autosummary::
  :toctree: generated/

  eig
  eigh
  eigvals
  eigvalsh
```

## Norms and other numbers
```{eval-rst}
.. autosummary::
  :toctree: generated/

  norm
  cond
  det
  matrix_rank
  slogdet
  trace
```

## Indexing
```{eval-rst}
.. autosummary::
  :toctree: generated/

  diagonal
```

## Solving equations and inverting matrices
```{eval-rst}
.. autosummary::
  :toctree: generated/

  solve
  inv
```

## Convenience functions

```{eval-rst}
.. autosummary::
  :toctree: generated/

  get_default_dims
```
