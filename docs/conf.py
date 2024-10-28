
project = "vectorgebra"
author = "Ahmet Erdem"
release = "4.0.0b1"

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

source_suffix = ".md"
master_doc = 'index'
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'collapse_navigation': True,
    'display_version': True,
    'logo_only': False,
}
