# API reference
`xarray_einstats` is structured in modules.
Each module contains wrappers to objects of a single library.

This keeps the library organized and eases handling of
optional dependencies: modules are not imported
when importing the library, only when importing the
module. Each module explains its optional dependencies
(if any) and if necessary gives specific installation advise.

:::{toctree}
:maxdepth: 1

stats
linalg
einops
:::
