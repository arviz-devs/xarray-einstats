(installation)=
# Installation

## Pip
You can install `xarray-eistats` with

::::{tab-set}
:::{tab-item} Stable
:sync: stable

```bash
pip install xarray-einstats
```
:::
:::{tab-item} Development
:sync: dev

```bash
pip install xarray-einstats @ git+https://github.com/arviz-devs/xarray-einstats
```
:::
::::

or, to install some of its optional dependencies, with

::::{tab-set}
:::{tab-item} Stable
:sync: stable

```bash
pip install "xarray-einstats[<option>]"
```
:::
:::{tab-item} Development
:sync: dev

```bash
pip install "xarray-einstats[<option>] @ git+https://github.com/arviz-devs/xarray-einstats"
```
:::
::::

you can install multiple bundles of optional dependencies separating them with commas.
Thus, to install all user facing optional dependencies you should use `xarray-einstats[einops,numba]`

After installation, independently of the command chosen,
you can import it with `import xarray_einstats`.

:::{caution}
Not all modules are loaded when importing the library to keep
some dependencies optional. Check the help of each module
to see what they need to work.
:::

Currently, optional dependency bundles include:

* `einops`
* `numba`
* `test` (for developers)
* `doc` (for developers)

## Conda
`xarray-einstats` is also available on [conda forge](https://anaconda.org/conda-forge/xarray-einstats). It can be installed with conda (or mamba) using:

```bash
conda install -c conda-forge xarray-einstats
```

Only published releases and the base package (no optional dependencies) can be installed with conda.
To use all features of `xarray-einstats` you will need to install the optional dependencies manually.
