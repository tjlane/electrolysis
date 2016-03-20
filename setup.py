#coding: utf8

"""
Setup script for electrolysis.
"""

from glob import glob


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(name='electrolysis',
      packages=["electrolysis", "electrolysis.inout"],
      package_dir={"electrolysis": "electrolysis",
                   "electrolysis.inout": "electrolysis/inout"},
      scripts=[s for s in glob('scripts/*') if not s.endswith('__.py')]
      )
