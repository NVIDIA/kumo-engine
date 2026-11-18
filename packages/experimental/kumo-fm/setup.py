import pyarrow
from setuptools import Extension, setup

setup(ext_modules=[Extension(
    "kumo.fm._native_json", ["src/kumo/fm/_native_json.cpp"],
    include_dirs=[pyarrow.get_include()], language="c++",
    extra_compile_args=["-std=c++17", "-O3"],
)])
