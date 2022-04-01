(installation)=
# Installation

You can install `xarray-eistats` with

::::{tab-set}
:::{tab-item} Stable
:sync: stable

```
pip install xarray-einstats
```
:::
:::{tab-item} Development
:sync: dev

```
pip install xarray-einstats @ git+https://github.com/arviz-devs/xarray-einstats
```
:::
::::

or, to install some of its optional dependencies, with

::::{tab-set}
:::{tab-item} Stable
:sync: stable

```
pip install "xarray-einstats[<option>]"
```
:::
:::{tab-item} Development
:sync: dev

```
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
