(about_einops)=
# About the einops module
The einops module is a bit particular because the wrapers
_adapt_ the call signature and syntax to xarray but
they don't extend it. The reductions and rearrangements you'll be able
to do with `xarray_einstats.einops` are the same you can do in einops.
What changes is the objects on which they work and how you specify
those operations.

Why change the notation? There are three main reasons, each concerning one
of the elements respectively: `->`, space as delimiter and parenthesis:
* In xarray dimensions are already labeled. In many cases, the left
  side in the einops notation is only used to label the dimensions.
  In fact, 5/7 examples in https://einops.rocks/api/rearrange/ fall in this category.
  This is not necessary when working with xarray objects.
* In xarray dimension names can be any {term}`hashable <xarray:name>`. `xarray-einstats` only
  supports strings as dimension names, but the space can't be used as delimiter.
* In xarray dimensions are labeled and the order doesn't matter.
  This might seem the same as the first reason but it is not. When splitting
  or stacking dimensions you need (and want) the names of both parent and children dimensions.
  In some cases, for example stacking, we can autogenerate a default name, but
  in general you'll want to give a name to the new dimension. After all,
  dimension order in xarray doesn't matter and there isn't much to be done without knowing
  the dimension names.

:::{attention}
We also provide some cruder wrappers with syntax closer to einops.
We are experimenting on trying to find the right spot between being clear,
semantic and flexible yet concise.

These `raw_` wrappers like {func}`xarray_einstats.einops.raw_rearrange`
impose several extra constraints to accepted xarray inputs, in addition
to dimension names being strings.

The example data we are using on this page uses single word alphabetical
dimensions names which allows us to demonstrate both side by side.
:::

The guiding principle of the einops module is to take the input expressions
in our list of str/list/dict and translate them to valid einops expressions
using placeholders given the actual dimension names might not be valid
in einops expressions. Moreover, given that we already have the dimensions
labeled, we can take advantage of that during the translation process
and thus support "partial" expressions that cover only the dimensions
that will be modified.

Another important consideration is to take into account that _in xarray_,
dimension names should not matter, hence the constraint of using dicts
on the left side. Imposing this constraint also
makes our job of filling in the "partial" expressions much easier.
We do accept that in the right side as we can generate sensible
default names.

As for the `raw_` wrappers, in order to avoid rewriting the partial
expression filling logic, their behaviour is very simplified:
1. Split the expression in two if possible using `->`
2. Convert each side to list of str/list/dict following the rules of the complete wrappers
3. Call the complete wrapper

This has an extra and a bit hidden advantage. einops supports
_explicit_ ellipsis but we don't, to us an ellipsis is not writing
the dimension name in the expression. Therefore, `.` are valid
in our `raw_` expressions, we convert those to "full xarray" expressions
which support everything and we don't need extra logic to handle dots either.
