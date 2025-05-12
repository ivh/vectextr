from setuptools import setup, Extension
import pybind11
import numpy as np

# Define the extension module
ext_modules = [
    Extension(
        'vectextr',  # Module name
        ['extract_wrapper.cpp', 'extract.c'],  # Source files
        include_dirs=[
            # Path to pybind11 headers
            pybind11.get_include(),
            # Path to numpy headers
            np.get_include(),
        ],
        language='c++',
        extra_compile_args=['-std=c++11'],
    ),
]

# Setup function
setup(
    name='vectextr',
    version='0.1',
    description='Python bindings for vectextr using pybind11',
    ext_modules=ext_modules,
    install_requires=['numpy'],
)
