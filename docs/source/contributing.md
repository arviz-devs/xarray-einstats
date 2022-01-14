# Contributing overview

## Before contributing

Welcome to xarray-einstats! Before contributing to the project,
make sure that you **read our {doc}`code of conduct <code_of_conduct>`**

## Contributing code

1. Set up a Python development environment
   (advice: use [venv](https://docs.python.org/3/library/venv.html),
   [virtualenv](https://virtualenv.pypa.io/), or [miniconda](https://docs.conda.io/en/latest/miniconda.html))
2. Install tox: `python -m pip install tox`
3. Clone the repository
4. Start a new branch off main: `git switch -c new-branch main`
5. Make your code changes
6. Check that your code follows the style guidelines of the project: `tox -e reformat && tox -e check`
7. Run the tests: `tox -e py39`
   (change the version number according to the Python you are using)
8. Commit, push, and open a pull request!
