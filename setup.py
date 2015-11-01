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
      packages=["electrolysis"],
      package_dir={"electrolysis": "electrolysis"})
