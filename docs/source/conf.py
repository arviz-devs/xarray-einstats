import sys
import os
import pathlib
from importlib.metadata import metadata


# import local version of library instead of installed one
sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve().parent.parent / "src"))

# -- Project information

_metadata = metadata("xarray-einstats")

project = _metadata["Name"]
author = _metadata["Author-email"].split("<", 1)[0].strip()
copyright = f"2022, {author}"

version = _metadata["Version"]
if os.environ.get("READTHEDOCS", False):
    rtd_version = os.environ.get("READTHEDOCS_VERSION", "")
    if "." not in rtd_version and rtd_version.lower() != "stable":
        version = "dev"
else:
    branch_name = os.environ.get("BUILD_SOURCEBRANCHNAME", "")
    if branch_name == "main":
        version = "dev"
release = version


# -- General configuration

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "numpydoc",
    "myst_nb",
    "sphinx_copybutton",
    "jupyter_sphinx",
    "sphinx_design",
]

templates_path = ["_templates"]

exclude_patterns = [
    "Thumbs.db",
    ".DS_Store",
    ".ipynb_checkpoints",
    "tutorials/einops-image.zarr",
    "**/*.py",
]

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "autolink"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

extlinks = {
    "issue": ("https://github.com/arviz-devs/xarray-einstats/issues/%s", "GH#%s"),
    "pull": ("https://github.com/arviz-devs/xarray-einstats/pull/%s", "PR#%s"),
}

# -- Options for extensions

nb_execution_mode = "auto"
nb_execution_excludepatterns = ["*.ipynb"]
nb_kernel_rgx_aliases = {".*": "python3"}
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]

autosummary_generate = True
autodoc_typehints = "none"
autodoc_default_options = {
    "members": False,
}

numpydoc_show_class_members = False
numpydoc_xref_param_type = True
numpydoc_xref_ignore = {"of", "or", "optional", "scalar"}
singulars = ("int", "list", "dict", "float")
numpydoc_xref_aliases = {
    "DataArray": ":class:`xarray.DataArray`",
    "Dataset": ":class:`xarray.Dataset`",
    "pattern_list": "list of str, list or dict",
    "DimHandler": ":class:`~xarray_einstats.einops.DimHandler`",
    **{f"{singular}s": f":any:`{singular}s <{singular}>`" for singular in singulars},
}


# -- Options for HTML output

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#0f718e",
        "color-brand-content": "#069fac",
    },
    "dark_css_variables": {
        "color-brand-primary": "#069fac",
        "color-brand-content": "#00c0bf",
    },
}

intersphinx_mapping = {
    "arviz": ("https://python.arviz.org/en/latest/", None),
    "arviz_org": ("https://www.arviz.org/en/latest/", None),
    "dask": ("https://docs.dask.org/en/latest/", None),
    "numba": ("https://numba.pydata.org/numba-doc/dev", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "xhistogram": ("https://xhistogram.readthedocs.io/en/latest", None),
}
