#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_oracle_native, module) {
    module.doc() = "Native offline oracle core";
    module.def("api_version", []() { return 1; });
}
