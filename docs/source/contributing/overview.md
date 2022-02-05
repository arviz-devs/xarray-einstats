# Contributing overview

## Before contributing

Welcome to xarray-einstats! Before contributing to the project,
make sure that you **read our {ref}`code of conduct <code_of_conduct>`**

## Contributing code

1. Set up a Python development environment
   (advice: use [venv](https://docs.python.org/3/library/venv.html),
   [virtualenv](https://virtualenv.pypa.io/), or [miniconda](https://docs.conda.io/en/latest/miniconda.html))
1. Install tox: `python -m pip install tox`
1. Clone the repository
1. Start a new branch off main: `git switch -c new-branch main`
1. Install the library in editable mode: `pip install -e .` (note: requires `pip>=21.3`)
1. Make your code changes (and try them as you work)
1. Check that your code follows the style guidelines of the project: `tox -e reformat && tox -e check`
1. (optional) Build the documentation with `tox -e docs`. Note that tox commands `cleandocs` and `viewdocs`, more details about each command in {ref}`dev_reference`
   are also available
1. (optional) Run the tests: `tox -e py39`
   (change the version number according to the Python you are using)
1. Commit, push, and open a pull request!
