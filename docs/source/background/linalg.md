(about_linalg)=
# About the linalg module
All the wrappers in this module (with the notable exception of `einsum`) are
short and simple wrappers. Most are <5 lines and focus on providing a
comfortable and accessible API, not on extending the capabilities of the
functions wrapped.

The skeleton of such wrappers follows this pattern:
1. If no dims are provided, try to get default values for them. This is the main "extension"
   provided by this module. Doing `inv(da, dims=["dim2", "dim2"])` still feels like
   too much unnecessary information is needed. We believe that in the majority of cases
   the dimensions corresponding to the matrices can be guessed with simple logic.
   Therefore, as explained in {ref}`about_linalg/default_dims` section we provide a way
   for users to define functions that set or infer the default matrix dimension names.
   We expect this to be particularly helpful in workflows that chain multiple
   operations on the same matrix and its results.
2. If required, set the dimensions of the output, generating new dimension names
   from the provided ones.
3. Call `xarray.apply_ufunc`

## einsum
`einsum` and `einsum_path` are the only ones that don't follow this pattern, and
they diverge significantly from it as we believe is required.

xarray already has an `einsum` wrapper {func}`xarray.dot`, but it has two important
limitations. It reduces all the dimensions provided (so it can't be used to
implement a matrix multiplication for example) and it takes a single list of dimensions,
common in for all inputs (you can't reduce the "dim1" dimension for the 1st input but not the 2nd).

This means that a lot of the capabilities of `einsum` are already covered by `xarray.dot` or
by the functions in the `einops` module. However, we believe that the cases that
are not covered (such as matrix multiplication or outer product)
are relevant enough to write this `einsum` wrapper.

It's logic is similar to that of the {ref}`einops module <about_einops>` but with different
syntax, now adapted to `einsum` instead of `einops`. Here multiple inputs are supported
but decomposition/stacking is not, so dimensions being a list of lists means multiple outputs,
not decomposition/stacking.

Moreover, most of the logic is abstracted into {func}`xarray_einstats.linalg._einsum_parent`
which enables us to support both `einsum` and `einsum_path` with very little extra code for
each. We hope that having `einsum_path` will be helpful and accelerate operations
for our users.

(about_linalg/default_dims)=
## Default dimensions
As we explained in the introduction, we believe that allowing users to define the logic
used to choose the dimensions on which to operate by default can be very powerful,
even a game changer.

After considering some alternatives, we ended up deciding on providing the infrastructure
and encouraging monkeypatching as a way for users to costumize this logic. We want
the functions in this module to have a simple call signature as close as possible
to their numpy counterpart. Using alternatives like inheritance would require restructuring
the module into a class and class methods. More functional approaches require
adding an argument to the functions, if we ask users to pass an extra argument then
it can be directly the dimensions on which to operate.
