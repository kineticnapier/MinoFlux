#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace {

struct Cell {
    int dx = 0;
    int dy = 0;
};

using Rotation = std::array<Cell, 4>;
using PieceShapes = std::array<Rotation, 4>;

std::unordered_map<char, PieceShapes> g_shapes;

void register_shapes(const py::dict& mapping) {
    std::unordered_map<char, PieceShapes> parsed;
    for (auto item : mapping) {
        const std::string piece = py::cast<std::string>(item.first);
        if (piece.size() != 1) {
            throw std::runtime_error("piece keys must be one character");
        }
        py::sequence rotations = py::reinterpret_borrow<py::sequence>(item.second);
        if (py::len(rotations) != 4) {
            throw std::runtime_error("every piece must have four rotations");
        }
        PieceShapes shapes{};
        for (int rotation = 0; rotation < 4; ++rotation) {
            py::sequence cells = py::reinterpret_borrow<py::sequence>(rotations[rotation]);
            if (py::len(cells) != 4) {
                throw std::runtime_error("every rotation must contain four cells");
            }
            for (int index = 0; index < 4; ++index) {
                py::sequence cell = py::reinterpret_borrow<py::sequence>(cells[index]);
                if (py::len(cell) != 2) {
                    throw std::runtime_error("shape cells must be (x, y) pairs");
                }
                shapes[static_cast<size_t>(rotation)][static_cast<size_t>(index)] = Cell{
                    py::cast<int>(cell[0]),
                    py::cast<int>(cell[1]),
                };
            }
        }
        parsed[piece[0]] = shapes;
    }
    g_shapes = std::move(parsed);
}

const PieceShapes& piece_shapes(char piece) {
    auto found = g_shapes.find(piece);
    if (found == g_shapes.end()) {
        throw std::runtime_error("native neural shapes have not been registered for this piece");
    }
    return found->second;
}

bool occupied_or_wall(
    const std::array<uint16_t, 64>& rows,
    int height,
    int width,
    int x,
    int y
) noexcept {
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return true;
    }
    return (rows[static_cast<size_t>(y)] & (uint16_t{1} << x)) != 0;
}

int classify_t_spin(
    const std::array<uint16_t, 64>& rows,
    int height,
    int width,
    char piece,
    int x,
    int y,
    int rotation,
    bool last_rotation,
    int kick_index
) noexcept {
    if (piece != 'T' || !last_rotation) {
        return 0;
    }
    const int pivot_x = x + 1;
    const int pivot_y = y + 1;
    const std::array<bool, 4> corners = {
        occupied_or_wall(rows, height, width, pivot_x - 1, pivot_y - 1),
        occupied_or_wall(rows, height, width, pivot_x + 1, pivot_y - 1),
        occupied_or_wall(rows, height, width, pivot_x - 1, pivot_y + 1),
        occupied_or_wall(rows, height, width, pivot_x + 1, pivot_y + 1),
    };
    const int corner_count = static_cast<int>(corners[0]) + static_cast<int>(corners[1]) +
        static_cast<int>(corners[2]) + static_cast<int>(corners[3]);
    if (corner_count < 3) {
        return 0;
    }
    static constexpr std::array<std::array<int, 2>, 4> front = {{
        {{0, 1}}, {{1, 3}}, {{2, 3}}, {{0, 2}},
    }};
    const auto pair = front[static_cast<size_t>(rotation & 3)];
    if ((corners[static_cast<size_t>(pair[0])] && corners[static_cast<size_t>(pair[1])]) || kick_index == 4) {
        return 2;  // full
    }
    return 1;  // mini
}

bool collides_spawn(
    const std::array<uint16_t, 64>& rows,
    int height,
    int width,
    char piece
) {
    const Rotation& shape = piece_shapes(piece)[0];
    constexpr int spawn_x = 3;
    constexpr int spawn_y = 1;
    for (const Cell& cell : shape) {
        const int x = spawn_x + cell.dx;
        const int y = spawn_y + cell.dy;
        if (x < 0 || x >= width || y >= height) {
            return true;
        }
        if (y >= 0 && (rows[static_cast<size_t>(y)] & (uint16_t{1} << x)) != 0) {
            return true;
        }
    }
    return false;
}

struct B2BOutcome {
    bool active = false;
    int chain = 0;
    int charge = 0;
    int attack_bonus = 0;
    int released = 0;
};

B2BOutcome resolve_b2b(
    bool active,
    int chain,
    bool difficult,
    int lines,
    bool perfect_clear
) noexcept {
    const bool was_active = active;
    const int current_chain = was_active ? std::max(0, chain) : 0;
    const bool cleared = std::max(0, lines) > 0;
    if (!cleared) {
        const int charge = was_active && current_chain >= 4 ? current_chain : 0;
        return B2BOutcome{was_active, current_chain, charge, 0, 0};
    }
    if (perfect_clear) {
        const int attack_bonus = was_active ? 1 : 0;
        const int next_chain = was_active ? current_chain + 2 : 2;
        const int charge = next_chain >= 4 ? next_chain : 0;
        return B2BOutcome{true, next_chain, charge, attack_bonus, 0};
    }
    if (difficult) {
        const int attack_bonus = was_active ? 1 : 0;
        const int next_chain = was_active ? current_chain + 1 : 0;
        const int charge = next_chain >= 4 ? next_chain : 0;
        return B2BOutcome{true, next_chain, charge, attack_bonus, 0};
    }
    const int released = was_active && current_chain >= 4 ? current_chain : 0;
    return B2BOutcome{false, 0, 0, 0, released};
}

int base_attack(int lines, int spin_kind) noexcept {
    const int count = std::max(0, lines);
    if (spin_kind == 1 && count == 1) {
        return 0;
    }
    if (spin_kind == 2 && count == 1) {
        return 2;
    }
    if (spin_kind != 0 && count == 2) {
        return 4;
    }
    if (spin_kind != 0 && count >= 3) {
        return 6;
    }
    switch (count) {
        case 2: return 1;
        case 3: return 2;
        case 4: return 4;
        default: return 0;
    }
}

float clip01(double value) noexcept {
    if (value < 0.0) {
        return 0.0F;
    }
    if (value > 1.0) {
        return 1.0F;
    }
    return static_cast<float>(value);
}

void append_float(std::string& output, size_t& offset, float value) {
    std::memcpy(output.data() + offset, &value, sizeof(float));
    offset += sizeof(float);
}

std::vector<float> float_prefix(const py::sequence& values) {
    std::vector<float> result;
    result.reserve(static_cast<size_t>(py::len(values)));
    for (py::handle value : values) {
        result.push_back(py::cast<float>(value));
    }
    return result;
}

py::tuple encode_placement_group(
    const py::sequence& source_rows_object,
    const py::sequence& placements,
    const std::string& piece_name,
    const std::string& next_piece_name,
    int width,
    int height,
    int hidden_rows,
    int combo_before,
    bool back_to_back_before,
    int b2b_chain_before,
    const py::sequence& normal_prefix_object,
    const py::sequence& locked_prefix_object
) {
    if (piece_name.size() != 1 || next_piece_name.size() != 1) {
        throw std::runtime_error("piece names must be one character");
    }
    if (width <= 0 || width > 16 || height <= 0 || height > 64) {
        throw std::runtime_error("native neural encoder supports width <= 16 and height <= 64");
    }
    if (py::len(source_rows_object) != height) {
        throw std::runtime_error("source row count does not match height");
    }
    const char piece = piece_name[0];
    const char next_piece = next_piece_name[0];
    const PieceShapes& shapes = piece_shapes(piece);
    (void)piece_shapes(next_piece);

    std::array<uint16_t, 64> source_rows{};
    for (int y = 0; y < height; ++y) {
        const auto value = py::cast<uint32_t>(source_rows_object[y]);
        if (value >= (uint32_t{1} << width)) {
            throw std::runtime_error("source row contains bits outside board width");
        }
        source_rows[static_cast<size_t>(y)] = static_cast<uint16_t>(value);
    }

    const std::vector<float> normal_prefix = float_prefix(normal_prefix_object);
    const std::vector<float> locked_prefix = float_prefix(locked_prefix_object);
    if (normal_prefix.size() != locked_prefix.size()) {
        throw std::runtime_error("normal and locked context prefixes must have equal length");
    }
    const size_t prefix_size = normal_prefix.size();
    const size_t count = static_cast<size_t>(py::len(placements));
    const size_t context_size = prefix_size + 9;
    std::string board_bytes(count * static_cast<size_t>(height) * static_cast<size_t>(width), '\0');
    std::string context_bytes(count * context_size * sizeof(float), '\0');
    char* board_dest = board_bytes.data();
    size_t context_offset = 0;
    const uint16_t full_row = static_cast<uint16_t>((uint32_t{1} << width) - 1U);

    for (size_t placement_index = 0; placement_index < count; ++placement_index) {
        py::handle placement = placements[static_cast<py::ssize_t>(placement_index)];
        const std::string placement_piece = py::cast<std::string>(placement.attr("piece"));
        if (placement_piece != piece_name) {
            throw std::runtime_error("placement piece does not match group piece");
        }
        const int x = py::cast<int>(placement.attr("x"));
        const int y = py::cast<int>(placement.attr("y"));
        const int rotation = py::cast<int>(placement.attr("rotation")) & 3;
        const bool last_rotation = py::cast<bool>(placement.attr("last_move_was_rotation"));
        py::object kick_object = py::reinterpret_borrow<py::object>(placement.attr("rotation_kick_index"));
        const int kick_index = kick_object.is_none() ? -1 : py::cast<int>(kick_object);

        const int spin_kind = classify_t_spin(
            source_rows,
            height,
            width,
            piece,
            x,
            y,
            rotation,
            last_rotation,
            kick_index
        );

        std::array<uint16_t, 64> placed_rows = source_rows;
        bool topped_out = false;
        for (const Cell& cell : shapes[static_cast<size_t>(rotation)]) {
            const int cell_x = x + cell.dx;
            const int cell_y = y + cell.dy;
            if (cell_y < 0 || cell_y >= height) {
                topped_out = true;
                continue;
            }
            if (cell_x < 0 || cell_x >= width) {
                throw std::runtime_error("legal placement contains a cell outside board width");
            }
            placed_rows[static_cast<size_t>(cell_y)] = static_cast<uint16_t>(
                placed_rows[static_cast<size_t>(cell_y)] | (uint16_t{1} << cell_x)
            );
        }

        int lines = 0;
        for (int row_y = 0; row_y < height; ++row_y) {
            lines += static_cast<int>(placed_rows[static_cast<size_t>(row_y)] == full_row);
        }
        std::array<uint16_t, 64> after_rows{};
        int write_y = lines;
        for (int row_y = 0; row_y < height; ++row_y) {
            const uint16_t row = placed_rows[static_cast<size_t>(row_y)];
            if (row != full_row) {
                after_rows[static_cast<size_t>(write_y++)] = row;
            }
        }

        bool perfect_clear = true;
        for (int row_y = 0; row_y < height; ++row_y) {
            if (after_rows[static_cast<size_t>(row_y)] != 0) {
                perfect_clear = false;
                break;
            }
        }
        const bool difficult = lines == 4 || (spin_kind != 0 && lines > 0);
        const B2BOutcome b2b = resolve_b2b(
            back_to_back_before,
            b2b_chain_before,
            difficult,
            lines,
            perfect_clear && lines > 0
        );
        const int combo = lines ? combo_before + 1 : -1;
        int sent = base_attack(lines, spin_kind) + b2b.attack_bonus;
        if (lines && combo > 0) {
            sent += std::min(4, combo / 2 + 1);
        }
        if (perfect_clear && lines) {
            sent += 10;
        }
        const int total_sent = sent + b2b.released;

        bool hidden_occupied = false;
        for (int row_y = 0; row_y < std::min(height, std::max(0, hidden_rows)); ++row_y) {
            if (after_rows[static_cast<size_t>(row_y)] != 0) {
                hidden_occupied = true;
                break;
            }
        }
        const bool locked_out = topped_out || hidden_occupied;
        const bool game_over = locked_out ? true : collides_spawn(after_rows, height, width, next_piece);
        const std::vector<float>& prefix = locked_out ? locked_prefix : normal_prefix;

        for (float value : prefix) {
            append_float(context_bytes, context_offset, value);
        }
        append_float(context_bytes, context_offset, clip01(static_cast<double>(combo + 1) / 16.0));
        append_float(context_bytes, context_offset, b2b.active ? 1.0F : 0.0F);
        append_float(context_bytes, context_offset, clip01(static_cast<double>(b2b.chain) / 20.0));
        append_float(context_bytes, context_offset, clip01(static_cast<double>(b2b.charge) / 20.0));
        append_float(context_bytes, context_offset, game_over ? 1.0F : 0.0F);
        append_float(context_bytes, context_offset, clip01(static_cast<double>(lines) / 4.0));
        append_float(context_bytes, context_offset, clip01(static_cast<double>(total_sent) / 20.0));
        append_float(context_bytes, context_offset, spin_kind != 0 ? 1.0F : 0.0F);
        append_float(context_bytes, context_offset, perfect_clear ? 1.0F : 0.0F);

        for (int row_y = 0; row_y < height; ++row_y) {
            const uint16_t mask = after_rows[static_cast<size_t>(row_y)];
            for (int board_x = 0; board_x < width; ++board_x) {
                *board_dest++ = static_cast<char>((mask >> board_x) & 1U);
            }
        }
    }

    return py::make_tuple(py::bytes(board_bytes), py::bytes(context_bytes));
}

}  // namespace

PYBIND11_MODULE(_neural_native, module) {
    module.doc() = "Native helpers for neural compact-state encoding";
    module.def("register_shapes", &register_shapes, py::arg("mapping"));
    module.def(
        "encode_placement_group",
        &encode_placement_group,
        py::arg("source_rows"),
        py::arg("placements"),
        py::arg("piece"),
        py::arg("next_piece"),
        py::arg("width"),
        py::arg("height"),
        py::arg("hidden_rows"),
        py::arg("combo"),
        py::arg("back_to_back"),
        py::arg("b2b_chain"),
        py::arg("normal_prefix"),
        py::arg("locked_prefix")
    );
}
