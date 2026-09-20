#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "reachability_core.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace minoflux::reachability::pybind_support {
namespace py = pybind11;

inline uint64_t read_u64_le(const char* ptr) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(static_cast<unsigned char>(ptr[i])) << (8 * i);
    }
    return value;
}

inline std::vector<int32_t> to_i32_vector(const py::sequence& values, size_t expected, const char* name) {
    if (static_cast<size_t>(py::len(values)) != expected) {
        throw std::runtime_error(std::string(name) + " length mismatch");
    }
    std::vector<int32_t> result;
    result.reserve(expected);
    for (py::handle value : values) result.push_back(py::cast<int32_t>(value));
    return result;
}

inline std::vector<uint8_t> to_u8_bytes(const py::bytes& values, size_t expected, const char* name) {
    std::string raw = values;
    if (raw.size() != expected) throw std::runtime_error(std::string(name) + " length mismatch");
    return std::vector<uint8_t>(raw.begin(), raw.end());
}

inline std::vector<Mask256> decode_masks(const py::bytes& values, size_t count, const char* name) {
    std::string raw = values;
    if (raw.size() != count * kMaskBytes) throw std::runtime_error(std::string(name) + " length mismatch");
    std::vector<Mask256> result(count);
    for (size_t i = 0; i < count; ++i) {
        const char* base = raw.data() + i * kMaskBytes;
        for (size_t limb = 0; limb < 4; ++limb) result[i].words[limb] = read_u64_le(base + limb * 8);
    }
    return result;
}

inline std::shared_ptr<Table> table_from_python(
    const std::string& piece,
    int width,
    int height,
    int x_min,
    int x_max,
    int x_count,
    int y_min,
    const py::sequence& state_x,
    const py::sequence& state_y,
    const py::sequence& left_state,
    const py::sequence& right_state,
    const py::sequence& down_state,
    const py::bytes& collision_invalid,
    const py::bytes& collision_masks,
    const py::bytes& geometry_invalid,
    const py::bytes& geometry_masks,
    const py::sequence& rotation_transitions
) {
    if (width <= 0 || width > 64 || height <= 0 || width * height > 256) {
        throw std::runtime_error("native reachability table exceeds supported board mask size");
    }
    const size_t state_count = static_cast<size_t>(py::len(state_x));
    if (static_cast<size_t>(py::len(rotation_transitions)) != state_count) {
        throw std::runtime_error("rotation transition state count mismatch");
    }
    auto table = std::make_shared<Table>();
    table->width = width;
    table->height = height;
    table->x_min = x_min;
    table->x_max = x_max;
    table->x_count = x_count;
    table->y_min = y_min;
    table->state_count = static_cast<int>(state_count);
    table->limb_count = (width * height + 63) / 64;
    table->piece_is_t = piece == "T";
    table->state_x = to_i32_vector(state_x, state_count, "state_x");
    table->state_y = to_i32_vector(state_y, state_count, "state_y");
    table->left_state = to_i32_vector(left_state, state_count, "left_state");
    table->right_state = to_i32_vector(right_state, state_count, "right_state");
    table->down_state = to_i32_vector(down_state, state_count, "down_state");
    table->collision_invalid = to_u8_bytes(collision_invalid, state_count, "collision_invalid");
    table->geometry_invalid = to_u8_bytes(geometry_invalid, state_count, "geometry_invalid");
    table->collision_masks = decode_masks(collision_masks, state_count, "collision_masks");
    table->geometry_masks = decode_masks(geometry_masks, state_count, "geometry_masks");
    finalize_table(*table);

    table->state_group_offsets.reserve(state_count + 1);
    table->state_group_offsets.push_back(0);
    table->group_kick_offsets.push_back(0);
    size_t groups_per_state = 0;
    bool uniform_group_count = true;
    bool first_state = true;
    for (py::handle state_groups_handle : rotation_transitions) {
        py::sequence state_groups = py::reinterpret_borrow<py::sequence>(state_groups_handle);
        const size_t state_group_count = static_cast<size_t>(py::len(state_groups));
        if (first_state) {
            groups_per_state = state_group_count;
            first_state = false;
        } else if (state_group_count != groups_per_state) {
            uniform_group_count = false;
        }
        for (py::handle kicks_handle : state_groups) {
            py::sequence kicks = py::reinterpret_borrow<py::sequence>(kicks_handle);
            for (py::handle kick_handle : kicks) {
                py::sequence kick = py::reinterpret_borrow<py::sequence>(kick_handle);
                if (py::len(kick) != 2) throw std::runtime_error("kick tuple must have two entries");
                table->kick_targets.push_back(py::cast<int32_t>(kick[0]));
                table->kick_indices.push_back(py::cast<int8_t>(kick[1]));
            }
            table->group_kick_offsets.push_back(static_cast<uint32_t>(table->kick_targets.size()));
        }
        table->state_group_offsets.push_back(static_cast<uint32_t>(table->group_kick_offsets.size() - 1));
    }

    bool compact_ok =
        uniform_group_count &&
        groups_per_state <= 0xffu &&
        state_count <= kCompactRotationStateLimit &&
        table->kick_targets.size() <= 0xffffu;
    if (compact_ok) {
        for (uint32_t offset : table->group_kick_offsets) {
            if (offset > 0xffffu) {
                compact_ok = false;
                break;
            }
        }
    }
    if (compact_ok) {
        table->compact_group_kick_offsets.reserve(table->group_kick_offsets.size());
        for (uint32_t offset : table->group_kick_offsets) {
            table->compact_group_kick_offsets.push_back(static_cast<uint16_t>(offset));
        }
        table->compact_kicks.reserve(table->kick_targets.size());
        for (size_t index = 0; index < table->kick_targets.size(); ++index) {
            const int32_t target = table->kick_targets[index];
            const int32_t kick = table->kick_indices[index];
            if (
                target < 0 ||
                static_cast<uint32_t>(target) >= kCompactRotationStateLimit ||
                kick < 0 ||
                kick > kKickIndexMask
            ) {
                compact_ok = false;
                break;
            }
            table->compact_kicks.push_back(static_cast<uint16_t>(
                (static_cast<uint32_t>(target) << kKickIndexBits) |
                static_cast<uint32_t>(kick)
            ));
        }
    }
    if (compact_ok) {
        table->rotation_groups_per_state = static_cast<uint8_t>(groups_per_state);
        table->compact_rotation = true;
    } else {
        table->compact_group_kick_offsets.clear();
        table->compact_kicks.clear();
    }
    return table;
}

inline std::vector<uint64_t> rows_from_python(const py::sequence& row_values, int width, int height) {
    if (static_cast<int>(py::len(row_values)) != height) throw std::runtime_error("board row count mismatch");
    std::vector<uint64_t> rows;
    rows.reserve(static_cast<size_t>(height));
    const uint64_t width_mask = width == 64 ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
    for (py::handle value : row_values) rows.push_back(py::cast<uint64_t>(value) & width_mask);
    return rows;
}

}  // namespace minoflux::reachability::pybind_support
