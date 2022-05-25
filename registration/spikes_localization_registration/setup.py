#!/usr/bin/env python

#  python3 setup.py build_ext --inplace
from setuptools import setup, Extension
from Cython.Build import cythonize

with open("requirements.txt") as f:
    require = [x.strip() for x in f.readlines() if not x.startswith("git+")]

ext_modules = cythonize(
    ["subtraction_pipeline/ibme_fast_raster.pyx"]
)



setup(
    name="subtraction_pipeline",
    setup_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        'setuptools>=18.0',
        'cython',
    ],
    install_requires=require,
    version="0.1",
    packages=["subtraction_pipeline"],
    ext_modules=ext_modules,
)
        # cf. https://stackoverflow.com/questions/37471313/setup-requires-with-cython
        # Extension(
        #     'ibme_fast_raster',
        #     sources=["subtraction_pipeline/ibme_fast_raster.pyx"],
        # ),
    # ],
# )