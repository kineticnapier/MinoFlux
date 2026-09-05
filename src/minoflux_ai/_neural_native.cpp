#include <pybind11/pybind11.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

py::bytes expand_row_masks(py::buffer row_masks, int width) {
    if (width <= 0 || width > 16) {
        throw std::runtime_error("width must be between 1 and 16");
    }
    py::buffer_info info = row_masks.request();
    if (info.ndim != 1 || info.itemsize != static_cast<py::ssize_t>(sizeof(uint16_t))) {
        throw std::runtime_error("row_masks must be a contiguous 1D uint16 buffer");
    }
    if (info.strides[0] != static_cast<py::ssize_t>(sizeof(uint16_t))) {
        throw std::runtime_error("row_masks must be contiguous");
    }

    const size_t count = static_cast<size_t>(info.shape[0]);
    std::string output(count * static_cast<size_t>(width), '\0');
    const auto* base = static_cast<const unsigned char*>(info.ptr);
    char* dest = output.data();
    for (size_t index = 0; index < count; ++index) {
        uint16_t mask = 0;
        std::memcpy(&mask, base + index * sizeof(uint16_t), sizeof(uint16_t));
        for (int x = 0; x < width; ++x) {
            *dest++ = static_cast<char>((mask >> x) & 1U);
        }
    }
    return py::bytes(output);
}

}  // namespace

PYBIND11_MODULE(_neural_native, module) {
    module.doc() = "Native helpers for neural compact-state encoding";
    module.def("expand_row_masks", &expand_row_masks, py::arg("row_masks"), py::arg("width") = 10);
}
