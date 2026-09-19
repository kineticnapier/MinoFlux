#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace minoflux::oracle {

constexpr int kWidth = 10;
constexpr int kHeight = 24;
constexpr int kHiddenRows = 4;
constexpr uint16_t kFullRow = (1u << kWidth) - 1u;

enum class Piece : int8_t {
    I = 0,
    O = 1,
    T = 2,
    S = 3,
    Z = 4,
    J = 5,
    L = 6,
    None = -1,
};

struct Move {
    Piece piece = Piece::None;
    int8_t x = 0;
    int8_t y = 0;
    int8_t rotation = 0;
    bool use_hold = false;
    bool last_rotation = false;
    int8_t kick_index = -1;
    int8_t rotation_from = -1;
    int8_t rotation_to = -1;
};

struct State {
    std::array<uint16_t, kHeight> rows{};
    Piece current = Piece::T;
    Piece hold = Piece::None;
    uint16_t queue_index = 0;
    int16_t combo = -1;
    uint16_t b2b_chain = 0;
    bool has_hold = false;
    bool can_hold = true;
    bool b2b_active = false;
    bool game_over = false;
};

struct Config {
    int beam_width = 2000;
    int depth = 18;
    bool allow_180 = true;
    int max_nodes = 8000;
};

struct Result {
    bool found = false;
    Move move{};
    double score = 0.0;
};

Piece piece_from_char(char value);
char piece_to_char(Piece piece);
std::vector<Move> reachable_moves(
    const std::array<uint16_t, kHeight>& rows,
    Piece piece,
    bool allow_180,
    int max_nodes,
    bool use_hold = false
);
Result search(const State& root, std::span<const Piece> queue, const Config& config);

}  // namespace minoflux::oracle
