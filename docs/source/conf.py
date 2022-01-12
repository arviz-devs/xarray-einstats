import sys
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
release = ".".join(version.split(".")[:2])


# -- General configuration

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "numpydoc",
    "myst_nb",
    "sphinx_copybutton",
]

templates_path = ["_templates"]


# -- Options for HTML output

html_theme = "furo"
html_static_path = ["_static"]

intersphinx_mapping = {
    "dask": ("https://docs.dask.org/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "xarray": ("http://xarray.pydata.org/en/stable/", None),
}

