# Configuration file for the Sphinx documentation builder.

import re

project = "impact-index"
copyright = "2024, Benjamin Piwowarski"
author = "Benjamin Piwowarski"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "autoapi.extension",
    "sphinx_codeautolink",
]

# -- AutoAPI configuration (reads .pyi stubs without importing the module) --
autoapi_type = "python"
autoapi_dirs = [".."]  # parent (python/), where impact_index.pyi lives
autoapi_file_patterns = ["*.pyi"]
# No generated module page: each guide page documents its own classes with
# autoapiclass directives (see _check_api_coverage below).
autoapi_generate_api_docs = False
autoapi_add_toctree_entry = False
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_python_class_content = "both"  # show both class docstring and __init__
autoapi_member_order = "groupwise"

# Napoleon settings (Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Intersphinx (cross-reference numpy, python stdlib)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- General configuration --
templates_path = ["_templates"]
exclude_patterns = ["_build"]

# -- HTML output --
html_theme = "furo"
html_title = "impact-index"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# GitHub banner (view/edit-on-GitHub links + icon in the sidebar)
html_theme_options = {
    "source_repository": "https://github.com/xpmir/impact-index/",
    "source_branch": "master",
    "source_directory": "python/docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/xpmir/impact-index",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 '
                "8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2."
                "01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53."
                "63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78"
                "-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 ."
                "67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2"
                ".2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 "
                "3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 "
                '8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            "class": "",
        },
    ],
}


# Docstrings come from Rust doc comments, which are Markdown: turn ```lang
# fences into RST code blocks so they render instead of breaking the page.
def _markdown_fences_to_rst(app, what, name, obj, options, lines):
    out, in_fence = [], False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_fence:
                lang = stripped[3:].split(",")[0].strip() or "text"
                out.extend([f".. code-block:: {lang}", ""])
            else:
                out.append("")
            in_fence = not in_fence
        elif in_fence:
            out.append("   " + line if line else "")
        else:
            out.append(line)
    lines[:] = out


# Every public name in the stubs must be documented on one of the guide
# pages, otherwise a new class would silently be missing from the docs.
def _check_api_coverage(app):
    import ast
    from pathlib import Path

    from sphinx.util import logging

    docs = Path(app.srcdir)
    tree = ast.parse((docs.parent / "impact_index.pyi").read_text())
    public = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "__all__" for t in node.targets)
    )
    documented = set()
    for rst in docs.glob("*.rst"):
        documented.update(
            re.findall(r"^\.\. autoapi\w+:: impact_index\.(\w+)", rst.read_text(), re.M)
        )
    for name in sorted(set(public) - documented):
        logging.getLogger(__name__).warning(
            "impact_index.%s is not documented on any page", name
        )


def setup(app):
    app.connect("autodoc-process-docstring", _markdown_fences_to_rst)
    app.connect("builder-inited", _check_api_coverage)
