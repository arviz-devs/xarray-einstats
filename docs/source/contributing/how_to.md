# Contributing How-to Guides
We recommend having already gone over the {ref}`contributing_overview`
page before reading the sections below.

## Running the test suite locally
This section covers how to run the test suite locally.
That is, without using `tox` (as recommended on the {ref}`contributing_overview`
or triggering CI by opening a pull request on GitHub.

You should follow these steps for example if you want to reproduce and fix a bug
that depends on the NumPy version and therefore need to test `xarray-einstats`
on an environment that has a specific set of dependencies installed.

1. Install xarray-einstats in editable mode in the desired environment:

   ```bash
   pip install -e ".[test,einops,numba]"
   ```

   This will install xarray-einstats in editable mode together with all optional
   dependencies and all testing related dependencies.

1. Run pytest

   ```bash
   pytest tests/ --cov --cov-report term
   ```

   This will run the tests in the tests suite and coverage analysis and
   print all the results on the terminal.

## Making a new release

### Release preparation
1. Create a new branch
1. Check dependency version pins in `pyproject.toml`, they should follow [SPEC 0](https://scientific-python.org/specs/spec-0000/) roughly.
1. Review the change log (`docs/source/changelog.md`). The unreleased section
   should be updated to the current version and release date _and not yet added_
1. Update the version number in `__init__.py`. That is, remove the `dev` flag, it should not
   be increased.
1. Rerun the notebooks in `docs/source/tutorials`
1. Open a PR, make sure docs build correctly and all tests pass.
   Once everything is green, merge the PR
1. Create a new release from GitHub, use as tag the version number prepended
   by `v`. i.e. `v0.1.0` or `v0.2.3`

### Post release tasks
1. Check the new version appears on the readthedocs version switcher. If it doesn't
   go to [readthedocs](https://readthedocs.org/projects/xarray-einstats/) and
   add it.
1. Bump the minor version, set the patch version to 0 and add the `dev` flag.
   It should look like `0.2.0.dev0`.
1. Update the [changelog](https://github.com/arviz-devs/xarray-einstats/blob/main/docs/source/changelog.md)
   to add the unreleased section back. Here is a template to copy paste:

   ```
   ## v0.x.x (Unreleased)
   ### New features

   ### Maintenance and fixes

   ### Documentation
   ```
1. Check conda-forge. It will automatically send a PR with the updated recipe which
   will need to be reviewed (and modified if dependencies were updated).

