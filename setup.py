from __future__ import annotations

import sys

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


oracle_compile_args = (
    ["/O2", "/Ob3", "/Oi", "/Ot"]
    if sys.platform == "win32"
    else ["-O3"]
)

oracle_dependencies = [
    "src/minoflux_ai/native/oracle_core.hpp",
    "src/minoflux_ai/native/oracle_search_core.cpp",
    "src/minoflux_ai/native/reachability_core.hpp",
    "src/minoflux_ai/native/reachability_pybind.hpp",
]


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
            depends=oracle_dependencies,
            optional=True,
        ),
    ],
    cmdclass={"build_ext": build_ext},
)
