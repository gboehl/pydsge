# -- Path setup --------------------------------------------------------------
import os
import sys
# autopep8: off
sys.path.insert(0, os.path.abspath(".."))
# must be called AFTER the above:
from pydsge import __version__
# autopep8: on

# -- Project information -----------------------------------------------------
project = "pydsge"
copyright = "2019, Gregor Boehl"
author = "Gregor Boehl"

version = __version__
release = version

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "IPython.sphinxext.ipython_console_highlighting",
]

templates_path = ["_templates", "**.ipynb_checkpoints"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_book_theme"
html_copy_source = True
html_show_sourcelink = True
html_sourcelink_suffix = ""
html_title = "pydsge"
html_static_path = ["_static"]
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/gboehl/pydsge",
    "repository_branch": "main",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "classic",
    },
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}

html_static_path = ["_static"]
autoclass_content = "both"
autodoc_member_order = "bysource"
master_doc = "index"
