# Contributing reference
Reference document for contributors.
Here we document part of the private API of the library as well as
the suite of developer commands and scripts configured and included in the repository.

## tox commands
`xarray-einstats` uses `tox` to automatically handle
testing (locally and on GitHub actions),
building documentation locally,
formatting and linting.

Here are the tox commands available which should be executed as
`tox -e <command>`

### reformat
Modifies the files using black and isort so they comply with formatting
requirements.

### check
Runs black, isort, pylint and pydocstyle to check both code style,
presence of documentation everywhere that follows numpydoc convention
and catching some code errors and bad practices.

### docs
Uses `sphinx-build` to generate the documentation.

### cleandocs
Deletes all doc cache and intermediate files to rebuild the docs from
scratch the next time you use the `docs` command.

### viewdocs
Uses `gnome-open` to open the documentation build by tox. Opens the homepage

### py3x
Runs test suite with pytest

## Private API

### Stats
```{eval-rst}
.. currentmodule:: xarray_einstats.stats
.. autosummary::
  :toctree: generated/


  _wrap_method
  _add_documented_method
  XrRV
  _apply_nonreduce_func
  _apply_reduce_func
```

### Linear Algebra
```{eval-rst}
.. currentmodule:: xarray_einstats.linalg
.. autosummary::
  :toctree: generated/

  PairHandler
  _einsum_parent
```

### Einops
```{eval-rst}
.. currentmodule:: xarray_einstats.einops
.. autosummary::
  :toctree: generated/

  DimHandler
  process_pattern_list
  translate_pattern
```

### Tutorial
```{eval-rst}
.. automodule:: xarray_einstats.tutorial
.. autosummary::
  :toctree: generated

  generate_mcmc_like_dataset
  generate_matrices_dataarray
```
