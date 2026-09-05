from __future__ import annotations

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


setup(
    ext_modules=[
        Pybind11Extension(
            "minoflux_ai._reachability_native",
            ["src/minoflux_ai/_reachability_native.cpp"],
            cxx_std=17,
            optional=True,
        )
    ],
    cmdclass={"build_ext": build_ext},
)
