# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime

import kumopql

author = 'Kumo.AI'
project = 'kumopql'
version = kumopql.__version__
copyright = f'{datetime.datetime.now().year}, {author}'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
    'sphinx_autodoc_typehints',
]

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 2,
}
templates_path = ['_templates']
autodoc_member_order = 'bysource'
copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True

intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/dev', None),
    'python': ('https://docs.python.org/3/', None),
    'PIL': ('https://pillow.readthedocs.io/en/stable/', None),
}


def rst_jinja_render(app, _, source):
    if hasattr(app.builder, 'templates'):
        rst_context = {'kumopql': kumopql}
        source[0] = app.builder.templates.render_string(source[0], rst_context)


def setup(app):
    app.connect('source-read', rst_jinja_render)
