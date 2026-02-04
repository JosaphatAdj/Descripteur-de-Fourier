"""
Setup script pour compiler l'extension Cython
"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
from pathlib import Path

# Chemins
project_root = Path(__file__).parent
c_src = project_root / "c_src"
lib_dir = project_root / "python" / "lib"

# Extension Cython
extensions = [
    Extension(
        "fourier_wrapper",
        sources=["python/fourier_wrapper.pyx"],
        include_dirs=[
            str(c_src / "include"),
            np.get_include()
        ],
        library_dirs=[str(lib_dir)],
        libraries=["fourier", "openblas", "m"],
        extra_compile_args=["-O3", "-march=native"],
        language="c"
    )
]

setup(
    name="fourier_wrapper",
    ext_modules=cythonize(extensions, 
                         compiler_directives={'language_level': "3"},
                         annotate=True),  # Génère un rapport HTML
)
