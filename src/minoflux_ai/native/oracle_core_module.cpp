#include "oracle_core.cpp"

namespace minoflux::oracle {

std::vector<Move> reachable_moves(
    const std::array<uint16_t, kHeight>& rows,
    Piece piece,
    bool allow_180,
    int max_nodes,
    bool use_hold
) {
    return reachable(rows, piece, allow_180, max_nodes, use_hold);
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

    if (move.piece != placing) {
        return result;
    }
    if (collides(state.rows, move.piece, move.x, move.y, move.rotation)) {
        return result;
    }
    if (!collides(state.rows, move.piece, move.x, move.y + 1, move.rotation)) {
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
    return result;
}

}  // namespace minoflux::oracle
