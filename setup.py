from __future__ import annotations

import sys

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


oracle_compile_args = (
    ["/O2", "/Ob3", "/Oi", "/Ot", "/arch:AVX2"]
    if sys.platform == "win32"
    else ["-O3"]
)


setup(
    ext_modules=[
        Pybind11Extension(
            "minoflux_ai._reachability_native",
            ["src/minoflux_ai/_reachability_native.cpp"],
            cxx_std=17,
            optional=True,
        ),
        Pybind11Extension(
            "minoflux_ai._neural_native",
            ["src/minoflux_ai/_neural_native.cpp"],
            cxx_std=17,
            optional=True,
        ),
        Pybind11Extension(
            "minoflux_ai._oracle_native",
            [
                "src/minoflux_ai/_oracle_native.cpp",
                "src/minoflux_ai/native/oracle_core_module.cpp",
            ],
            cxx_std=20,
            extra_compile_args=oracle_compile_args,
            optional=True,
        ),
    ],
    cmdclass={"build_ext": build_ext},
)
