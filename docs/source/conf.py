import sys
from os.path import abspath

sys.path.insert(0, abspath('../../'))
project = "vectorgebra"
author = "Ahmet Erdem"
release = "4.0.0b6"

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

autodoc_member_order = 'bysource'

source_suffix = ".rst"
master_doc = 'index'
