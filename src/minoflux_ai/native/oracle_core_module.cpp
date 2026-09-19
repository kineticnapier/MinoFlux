#include "oracle_search_core.cpp"

#include <array>
#include <memory>
#include <stdexcept>
#include <vector>

namespace minoflux::oracle {
namespace {
namespace reach = minoflux::reachability;

std::array<std::shared_ptr<const reach::Table>, 14> g_reachability_tables{};

size_t table_index(Piece piece, bool allow_180) {
    const int value = static_cast<int>(piece);
    if (value < 0 || value >= 7) {
        throw std::runtime_error("invalid oracle piece for reachability table");
    }
    return static_cast<size_t>(value * 2 + (allow_180 ? 1 : 0));
}

const reach::Table& table_for(Piece piece, bool allow_180) {
    const auto& table = g_reachability_tables[table_index(piece, allow_180)];
    if (!table) {
        throw std::runtime_error("oracle reachability table is not registered");
    }
    return *table;
}
}  // namespace

void register_reachability_table(
    Piece piece,
    bool allow_180,
    std::shared_ptr<const minoflux::reachability::Table> table
) {
    if (!table) {
        throw std::runtime_error("oracle reachability table is null");
    }
    if (table->width != kWidth || table->height != kHeight) {
        throw std::runtime_error("oracle reachability table dimensions do not match");
    }
    g_reachability_tables[table_index(piece, allow_180)] = std::move(table);
}

std::vector<Move> reachable_moves(
    const std::array<uint16_t, kHeight>& rows,
    Piece piece,
    bool allow_180,
    int max_nodes,
    bool use_hold
) {
    const reach::Table& table = table_for(piece, allow_180);
    std::vector<uint64_t> native_rows;
    native_rows.reserve(kHeight);
    for (uint16_t row : rows) {
        native_rows.push_back(row);
    }
    const reach::RunResult native_result = reach::run(
        table,
        native_rows,
        3,
        1,
        0,
        max_nodes,
        false
    );

    std::vector<Move> result;
    result.reserve(native_result.placements.size());
    for (const reach::PlacementRecord& placement : native_result.placements) {
        Move move;
        move.piece = piece;
        move.x = static_cast<int8_t>(placement.x);
        move.y = static_cast<int8_t>(placement.y);
        move.rotation = static_cast<int8_t>(placement.rotation);
        move.use_hold = use_hold;
        move.last_rotation = placement.last_rotation;
        move.kick_index = static_cast<int8_t>(placement.kick_index);
        move.rotation_from = static_cast<int8_t>(placement.rotation_from);
        move.rotation_to = static_cast<int8_t>(placement.rotation_to);
        result.push_back(move);
    }
    return result;
}

TransitionResult transition(
    const State& root,
    std::span<const Piece> queue,
    const Move& move
) {
    TransitionResult result;
    if (root.game_over || root.current == Piece::None) {
        return result;
    }

    State state = root;
    Piece placing = state.current;
    if (move.use_hold) {
        if (!state.can_hold) {
            return result;
        }
        const Piece outgoing = state.current;
        if (state.has_hold) {
            placing = state.hold;
        } else {
            if (state.queue_index >= queue.size()) {
                return result;
            }
            placing = queue[state.queue_index++];
            if (placing == Piece::None) {
                return result;
            }
        }
        state.hold = outgoing;
        state.has_hold = true;
        state.can_hold = false;
        state.current = placing;
        if (collides(state.rows, placing, 3, 1, 0)) {
            return result;
        }
    }

    if (move.piece != placing ||
        collides(state.rows, move.piece, move.x, move.y, move.rotation) ||
        !collides(state.rows, move.piece, move.x, move.y + 1, move.rotation)) {
        return result;
    }

    const int spin_kind =
        (move.piece == Piece::T && move.last_rotation)
            ? classify_t_spin(state.rows, move.x, move.y, move.rotation, move.kick_index)
            : 0;
    const bool was_active = state.b2b_active;
    const int previous_chain = was_active ? state.b2b_chain : 0;
    const Features parent_features = features(state.rows);
    LockValue value;
    apply_placement(state, queue, move, parent_features.holes, value);

    const int event = spin_event(spin_kind, value.lines);
    const bool difficult = difficult_clear(value.lines, event);
    const int released =
        (value.lines > 0 && !value.pc && !difficult && was_active && previous_chain >= 4)
            ? previous_chain
            : 0;

    result.valid = true;
    result.state = state;
    result.lines = value.lines;
    result.attack = value.attack;
    result.spin_event = event;
    result.perfect_clear = value.pc;
    result.surge_released = released;
    result.surge_charge = state.b2b_active && state.b2b_chain >= 4 ? state.b2b_chain : 0;
    return result;
}

}  // namespace minoflux::oracle
