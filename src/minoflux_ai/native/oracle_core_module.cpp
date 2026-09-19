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

}  // namespace minoflux::oracle
