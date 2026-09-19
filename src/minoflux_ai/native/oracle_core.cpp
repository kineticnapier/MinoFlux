#include "oracle_core.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace minoflux::oracle {
namespace {

struct Cell { int8_t x; int8_t y; };
struct Kick { int8_t x; int8_t y; };
struct KickSet { std::array<Kick, 6> values{}; uint8_t count = 0; };

constexpr Cell C(int x, int y) { return Cell{static_cast<int8_t>(x), static_cast<int8_t>(y)}; }
constexpr Kick K(int x, int y) { return Kick{static_cast<int8_t>(x), static_cast<int8_t>(y)}; }

using Shape = std::array<Cell, 4>;
using Rotations = std::array<Shape, 4>;

constexpr std::array<Rotations, 7> kShapes = {
    Rotations{
        Shape{C(0,1),C(1,1),C(2,1),C(3,1)},
        Shape{C(2,0),C(2,1),C(2,2),C(2,3)},
        Shape{C(0,2),C(1,2),C(2,2),C(3,2)},
        Shape{C(1,0),C(1,1),C(1,2),C(1,3)},
    },
    Rotations{
        Shape{C(1,0),C(2,0),C(1,1),C(2,1)},
        Shape{C(1,0),C(2,0),C(1,1),C(2,1)},
        Shape{C(1,0),C(2,0),C(1,1),C(2,1)},
        Shape{C(1,0),C(2,0),C(1,1),C(2,1)},
    },
    Rotations{
        Shape{C(1,0),C(0,1),C(1,1),C(2,1)},
        Shape{C(1,0),C(1,1),C(2,1),C(1,2)},
        Shape{C(0,1),C(1,1),C(2,1),C(1,2)},
        Shape{C(1,0),C(0,1),C(1,1),C(1,2)},
    },
    Rotations{
        Shape{C(1,0),C(2,0),C(0,1),C(1,1)},
        Shape{C(1,0),C(1,1),C(2,1),C(2,2)},
        Shape{C(1,1),C(2,1),C(0,2),C(1,2)},
        Shape{C(0,0),C(0,1),C(1,1),C(1,2)},
    },
    Rotations{
        Shape{C(0,0),C(1,0),C(1,1),C(2,1)},
        Shape{C(2,0),C(1,1),C(2,1),C(1,2)},
        Shape{C(0,1),C(1,1),C(1,2),C(2,2)},
        Shape{C(1,0),C(0,1),C(1,1),C(0,2)},
    },
    Rotations{
        Shape{C(0,0),C(0,1),C(1,1),C(2,1)},
        Shape{C(1,0),C(2,0),C(1,1),C(1,2)},
        Shape{C(0,1),C(1,1),C(2,1),C(2,2)},
        Shape{C(1,0),C(1,1),C(0,2),C(1,2)},
    },
    Rotations{
        Shape{C(2,0),C(0,1),C(1,1),C(2,1)},
        Shape{C(1,0),C(1,1),C(1,2),C(2,2)},
        Shape{C(0,1),C(1,1),C(2,1),C(0,2)},
        Shape{C(0,0),C(1,0),C(1,1),C(1,2)},
    },
};

constexpr KickSet ks1(Kick a) { return KickSet{{a},1}; }
constexpr KickSet ks2(Kick a, Kick b) { return KickSet{{a,b},2}; }
constexpr KickSet ks5(Kick a, Kick b, Kick c, Kick d, Kick e) { return KickSet{{a,b,c,d,e},5}; }
constexpr KickSet ks6(Kick a, Kick b, Kick c, Kick d, Kick e, Kick f) { return KickSet{{a,b,c,d,e,f},6}; }

KickSet kick_set(Piece piece, int from, int to) {
    from &= 3;
    to &= 3;
    if (piece == Piece::O) return ks1(K(0,0));
    const int delta = (to - from) & 3;
    if (delta == 2) {
        if (piece == Piece::I) {
            switch (from) {
                case 0: return ks2(K(0,0),K(0,-1));
                case 1: return ks2(K(0,0),K(1,0));
                case 2: return ks2(K(0,0),K(0,1));
                default:return ks2(K(0,0),K(-1,0));
            }
        }
        switch (from) {
            case 0: return ks6(K(0,0),K(0,-1),K(1,-1),K(-1,-1),K(1,0),K(-1,0));
            case 1: return ks6(K(0,0),K(1,0),K(1,-2),K(1,-1),K(0,-2),K(0,-1));
            case 2: return ks6(K(0,0),K(0,1),K(-1,1),K(1,1),K(-1,0),K(1,0));
            default:return ks6(K(0,0),K(-1,0),K(-1,-2),K(-1,-1),K(0,-2),K(0,-1));
        }
    }
    const int key = (from << 2) | to;
    if (piece == Piece::I) {
        switch (key) {
            case 0x1: return ks5(K(0,0),K(1,0),K(-2,0),K(-2,1),K(1,-2));
            case 0x6: return ks5(K(0,0),K(-1,0),K(2,0),K(-1,-2),K(2,1));
            case 0xb: return ks5(K(0,0),K(2,0),K(-1,0),K(2,-1),K(-1,2));
            case 0xc: return ks5(K(0,0),K(1,0),K(-2,0),K(1,2),K(-2,-1));
            case 0x3: return ks5(K(0,0),K(-1,0),K(2,0),K(2,1),K(-1,-2));
            case 0x4: return ks5(K(0,0),K(-1,0),K(2,0),K(-1,2),K(2,-1));
            case 0x9: return ks5(K(0,0),K(-2,0),K(1,0),K(-2,-1),K(1,2));
            case 0xe: return ks5(K(0,0),K(1,0),K(-2,0),K(1,-2),K(-2,1));
            default: break;
        }
    } else {
        switch (key) {
            case 0x1: return ks5(K(0,0),K(-1,0),K(-1,-1),K(0,2),K(-1,2));
            case 0x4: return ks5(K(0,0),K(1,0),K(1,1),K(0,-2),K(1,-2));
            case 0x6: return ks5(K(0,0),K(1,0),K(1,1),K(0,-2),K(1,-2));
            case 0x9: return ks5(K(0,0),K(-1,0),K(-1,-1),K(0,2),K(-1,2));
            case 0xb: return ks5(K(0,0),K(1,0),K(1,-1),K(0,2),K(1,2));
            case 0xe: return ks5(K(0,0),K(-1,0),K(-1,1),K(0,-2),K(-1,-2));
            case 0xc: return ks5(K(0,0),K(-1,0),K(-1,1),K(0,-2),K(-1,-2));
            case 0x3: return ks5(K(0,0),K(1,0),K(1,-1),K(0,2),K(1,2));
            default: break;
        }
    }
    throw std::runtime_error("invalid rotation transition");
}

constexpr int kXMin = -4;
constexpr int kXMax = 13;
constexpr int kYMin = -4;
constexpr int kXCount = kXMax - kXMin + 1;
constexpr int kYCount = kHeight - kYMin;
constexpr int kStateCount = kXCount * kYCount * 4;

int state_id(int x, int y, int r) noexcept {
    if (x < kXMin || x > kXMax || y < kYMin || y >= kHeight) return -1;
    return ((((y - kYMin) * kXCount) + (x - kXMin)) << 2) | (r & 3);
}

void decode_state(int id, int& x, int& y, int& r) noexcept {
    r = id & 3;
    const int flat = id >> 2;
    x = (flat % kXCount) + kXMin;
    y = (flat / kXCount) + kYMin;
}

bool collides(const std::array<uint16_t,kHeight>& rows, Piece piece, int x, int y, int rotation) noexcept {
    const auto& cells = kShapes[static_cast<size_t>(piece)][static_cast<size_t>(rotation & 3)];
    for (const Cell cell : cells) {
        const int cx = x + cell.x;
        const int cy = y + cell.y;
        if (cx < 0 || cx >= kWidth || cy >= kHeight) return true;
        if (cy >= 0 && (rows[static_cast<size_t>(cy)] & (uint16_t{1} << cx))) return true;
    }
    return false;
}

bool occupied_or_wall(const std::array<uint16_t,kHeight>& rows, int x, int y) noexcept {
    if (x < 0 || x >= kWidth || y < 0 || y >= kHeight) return true;
    return (rows[static_cast<size_t>(y)] & (uint16_t{1} << x)) != 0;
}

int classify_t_spin(const std::array<uint16_t,kHeight>& rows, int x, int y, int rotation, int kick_index) noexcept {
    const int px = x + 1;
    const int py = y + 1;
    const std::array<bool,4> corners = {
        occupied_or_wall(rows,px-1,py-1), occupied_or_wall(rows,px+1,py-1),
        occupied_or_wall(rows,px-1,py+1), occupied_or_wall(rows,px+1,py+1)
    };
    const int count = int(corners[0])+int(corners[1])+int(corners[2])+int(corners[3]);
    if (count < 3) return 0;
    constexpr std::array<std::array<int,2>,4> front = {{{{0,1}},{{1,3}},{{2,3}},{{0,2}}}};
    const auto p = front[static_cast<size_t>(rotation&3)];
    if ((corners[p[0]] && corners[p[1]]) || kick_index == 4) return 2;
    return 1;
}

uint32_t geometry_key(Piece piece, int x, int y, int rotation) noexcept {
    std::array<uint8_t,4> ids{};
    size_t n = 0;
    const auto& cells = kShapes[static_cast<size_t>(piece)][static_cast<size_t>(rotation&3)];
    for (const Cell cell : cells) {
        const int cx = x + cell.x;
        const int cy = y + cell.y;
        if (cx < 0 || cx >= kWidth || cy < 0 || cy >= kHeight) return std::numeric_limits<uint32_t>::max();
        ids[n++] = static_cast<uint8_t>(cy*kWidth+cx);
    }
    std::sort(ids.begin(),ids.end());
    return uint32_t(ids[0]) | (uint32_t(ids[1])<<8) | (uint32_t(ids[2])<<16) | (uint32_t(ids[3])<<24);
}

struct ReachCandidate {
    Move move{};
    int depth = 0;
    int order = 0;
    int spin = 0;
};

bool better_reach(const ReachCandidate& a, const ReachCandidate& b, bool is_t) noexcept {
    if (is_t && a.spin != b.spin) {
        const int aa = a.spin != 0;
        const int bb = b.spin != 0;
        if (aa != bb) return aa > bb;
        const int af = a.spin == 2;
        const int bf = b.spin == 2;
        if (af != bf) return af > bf;
    }
    if (a.depth != b.depth) return a.depth < b.depth;
    return a.order < b.order;
}

std::vector<Move> reachable(const std::array<uint16_t,kHeight>& rows, Piece piece, bool allow180, int max_nodes, bool use_hold) {
    if (piece == Piece::None || collides(rows,piece,3,1,0)) return {};
    std::array<int16_t,kStateCount> depth{};
    std::array<int16_t,kStateCount> kick_info{};
    std::array<int16_t,kStateCount> rot_depth{};
    std::array<int16_t,kStateCount> rot_kick{};
    std::array<uint8_t,kStateCount> rot_is_geometry{};
    depth.fill(-1); kick_info.fill(-1); rot_depth.fill(-1); rot_kick.fill(-1);
    std::vector<int> frontier; frontier.reserve(kStateCount);
    std::vector<int> visited; visited.reserve(kStateCount);
    std::vector<int> rot_visited; rot_visited.reserve(kStateCount);
    const int start = state_id(3,1,0);
    depth[static_cast<size_t>(start)] = 0;
    frontier.push_back(start); visited.push_back(start);
    size_t cursor = 0;
    int reachable_count = 1;
    const int budget = std::max(1,max_nodes);
    while (cursor < frontier.size() && reachable_count <= budget) {
        const int id = frontier[cursor++];
        int x,y,r; decode_state(id,x,y,r);
        const int nd = depth[static_cast<size_t>(id)] + 1;
        const std::array<std::pair<int,int>,3> moves = {{{x-1,y},{x+1,y},{x,y+1}}};
        for (const auto& [nx,ny] : moves) {
            const int target = state_id(nx,ny,r);
            if (target < 0 || depth[static_cast<size_t>(target)] >= 0) continue;
            if (!collides(rows,piece,nx,ny,r)) {
                depth[static_cast<size_t>(target)] = static_cast<int16_t>(nd);
                kick_info[static_cast<size_t>(target)] = -1;
                frontier.push_back(target); visited.push_back(target); ++reachable_count;
            }
        }
        if (piece != Piece::O) {
            const std::array<int,3> directions = {1,-1,2};
            const int direction_count = allow180 ? 3 : 2;
            for (int di=0; di<direction_count; ++di) {
                const int target_r = (r + directions[static_cast<size_t>(di)] + 4) & 3;
                const KickSet kicks = kick_set(piece,r,target_r);
                int target = -1;
                int used_kick = -1;
                for (int ki=0; ki<kicks.count; ++ki) {
                    const auto k = kicks.values[static_cast<size_t>(ki)];
                    const int nx=x+k.x, ny=y+k.y;
                    const int candidate=state_id(nx,ny,target_r);
                    if (candidate < 0) continue;
                    if (!collides(rows,piece,nx,ny,target_r)) { target=candidate; used_kick=ki; break; }
                }
                if (target < 0) continue;
                const bool adds = depth[static_cast<size_t>(target)] < 0;
                const bool improves_rot = piece == Piece::T && (rot_depth[static_cast<size_t>(target)] < 0 || nd < rot_depth[static_cast<size_t>(target)]);
                const int info = (r << 3) | used_kick;
                if (improves_rot) {
                    if (rot_depth[static_cast<size_t>(target)] < 0) rot_visited.push_back(target);
                    rot_depth[static_cast<size_t>(target)] = static_cast<int16_t>(nd);
                    rot_kick[static_cast<size_t>(target)] = static_cast<int16_t>(info);
                    rot_is_geometry[static_cast<size_t>(target)] = static_cast<uint8_t>(adds);
                }
                if (adds) {
                    depth[static_cast<size_t>(target)] = static_cast<int16_t>(nd);
                    kick_info[static_cast<size_t>(target)] = static_cast<int16_t>(info);
                    frontier.push_back(target); visited.push_back(target); ++reachable_count;
                }
            }
        }
        if (reachable_count > budget) break;
    }

    std::unordered_map<uint32_t,ReachCandidate> best;
    best.reserve(128);
    auto emit = [&](int id, int d, int info, int order) {
        int x,y,r; decode_state(id,x,y,r);
        int landing_y = y;
        while (!collides(rows,piece,x,landing_y+1,r)) ++landing_y;
        const uint32_t key = geometry_key(piece,x,landing_y,r);
        if (key == std::numeric_limits<uint32_t>::max()) return;
        ReachCandidate c;
        c.depth=d; c.order=order;
        c.move.piece=piece; c.move.x=static_cast<int8_t>(x); c.move.y=static_cast<int8_t>(landing_y); c.move.rotation=static_cast<int8_t>(r);
        c.move.use_hold=use_hold; c.move.last_rotation=info>=0;
        c.move.kick_index=info>=0 ? static_cast<int8_t>(info&7) : -1;
        c.move.rotation_from=info>=0 ? static_cast<int8_t>(info>>3) : -1;
        c.move.rotation_to=info>=0 ? static_cast<int8_t>(r) : -1;
        if (piece==Piece::T && info>=0) c.spin=classify_t_spin(rows,x,landing_y,r,c.move.kick_index);
        auto it=best.find(key);
        if (it==best.end() || better_reach(c,it->second,piece==Piece::T)) best[key]=c;
    };
    for (int id:visited) emit(id,depth[static_cast<size_t>(id)],kick_info[static_cast<size_t>(id)],id);
    if (piece==Piece::T) {
        for (int id:rot_visited) {
            if (rot_is_geometry[static_cast<size_t>(id)]) continue;
            emit(id,rot_depth[static_cast<size_t>(id)],rot_kick[static_cast<size_t>(id)],kStateCount+id);
        }
    }
    std::vector<Move> result; result.reserve(best.size());
    for (const auto& entry:best) result.push_back(entry.second.move);
    std::sort(result.begin(),result.end(),[](const Move& a,const Move& b){
        if (a.rotation!=b.rotation) return a.rotation<b.rotation;
        if (a.x!=b.x) return a.x<b.x;
        return a.y<b.y;
    });
    return result;
}

struct Features {
    int aggregate_height=0;
    int max_height=0;
    int holes=0;
    int hole_depth=0;
    int bumpiness=0;
    int wells=0;
    int t_spin_slots=0;
};

bool cell_occupied(const std::array<uint16_t,kHeight>& rows,int x,int y) noexcept {
    if (x<0 || x>=kWidth || y<0 || y>=kHeight) return true;
    return (rows[static_cast<size_t>(y)] & (uint16_t{1}<<x)) != 0;
}
bool cell_empty(const std::array<uint16_t,kHeight>& rows,int x,int y) noexcept {
    return x>=0 && x<kWidth && y>=0 && y<kHeight && !cell_occupied(rows,x,y);
}

Features features(const std::array<uint16_t,kHeight>& rows) {
    Features f;
    std::array<int,kWidth> heights{};
    for (int x=0;x<kWidth;++x) {
        int top=-1;
        int occupied_above=0;
        for (int y=0;y<kHeight;++y) {
            const bool occ=(rows[static_cast<size_t>(y)]&(uint16_t{1}<<x))!=0;
            if (occ) { if (top<0) top=y; ++occupied_above; }
            else if (top>=0) { ++f.holes; f.hole_depth += occupied_above; }
        }
        heights[static_cast<size_t>(x)] = top<0 ? 0 : kHeight-top;
    }
    for (int h:heights) f.aggregate_height += h;
    f.max_height=*std::max_element(heights.begin(),heights.end());
    for (int x=0;x<kWidth-1;++x) f.bumpiness += std::abs(heights[static_cast<size_t>(x)]-heights[static_cast<size_t>(x+1)]);
    for (int x=0;x<kWidth;++x) {
        int run=0;
        for (int y=0;y<kHeight;++y) {
            const bool well = !cell_occupied(rows,x,y) && (x==0 || cell_occupied(rows,x-1,y)) && (x==kWidth-1 || cell_occupied(rows,x+1,y));
            if (well) ++run;
            else { f.wells += run*(run+1)/2; run=0; }
        }
        f.wells += run*(run+1)/2;
    }
    for (int y=0;y<kHeight;++y) for (int x=0;x<kWidth;++x) {
        if (cell_occupied(rows,x,y)) continue;
        int corners = int(cell_occupied(rows,x-1,y-1))+int(cell_occupied(rows,x+1,y-1))+int(cell_occupied(rows,x-1,y+1))+int(cell_occupied(rows,x+1,y+1));
        if (corners<3) continue;
        int cardinals = int(cell_empty(rows,x,y-1))+int(cell_empty(rows,x,y+1))+int(cell_empty(rows,x-1,y))+int(cell_empty(rows,x+1,y));
        if (cardinals>=3) ++f.t_spin_slots;
    }
    return f;
}

double board_score(const Features& f, bool game_over) noexcept {
    const double density = double(f.t_spin_slots)/(1.0+double(f.holes));
    double score =
        f.aggregate_height * -0.510066 +
        f.max_height * -0.080000 +
        f.holes * -0.800000 +
        f.hole_depth * -0.120000 +
        f.bumpiness * -0.184483 +
        f.wells * -0.060000 +
        f.t_spin_slots * 1.200000 +
        density * 0.280000;
    if (game_over) score -= 1'000'000.0;
    return score;
}

int spin_event(int kind,int lines) noexcept {
    if (!kind) return 0;
    if (kind==1) {
        if (lines==0) return 1;
        if (lines==1) return 2;
        return lines==2 ? 5 : 6;
    }
    if (lines==0) return 3;
    if (lines==1) return 4;
    if (lines==2) return 5;
    return 6;
}
int base_attack(int lines,int event) noexcept {
    if (event==2) return 0;
    if (event==4) return 2;
    if (event==5) return 4;
    if (event==6) return 6;
    switch(lines){case 2:return 1;case 3:return 2;case 4:return 4;default:return 0;}
}
bool difficult_clear(int lines,int event) noexcept { return lines==4 || (event!=0 && lines>0); }

struct LockValue { int lines=0; int attack=0; int spin_lines=0; bool pc=false; int new_holes=0; };

bool spawn_next(State& state,std::span<const Piece> queue) noexcept {
    if (state.queue_index >= queue.size()) return false;
    state.current = queue[state.queue_index++];
    state.can_hold = true;
    return state.current != Piece::None && !collides(state.rows,state.current,3,1,0);
}

bool apply_placement(State& state,std::span<const Piece> queue,const Move& move,int parent_holes,LockValue& value) {
    const int spin_kind = (move.piece==Piece::T && move.last_rotation) ? classify_t_spin(state.rows,move.x,move.y,move.rotation,move.kick_index) : 0;
    bool topped=false;
    const auto& cells=kShapes[static_cast<size_t>(move.piece)][static_cast<size_t>(move.rotation&3)];
    for (const Cell c:cells) {
        const int x=move.x+c.x, y=move.y+c.y;
        if (y<0) topped=true;
        else state.rows[static_cast<size_t>(y)] |= uint16_t{1}<<x;
    }
    std::array<uint16_t,kHeight> cleared{};
    int write=kHeight-1;
    int lines=0;
    for (int y=kHeight-1;y>=0;--y) {
        if (state.rows[static_cast<size_t>(y)]==kFullRow) ++lines;
        else cleared[static_cast<size_t>(write--)]=state.rows[static_cast<size_t>(y)];
    }
    state.rows=cleared;
    const bool pc=std::all_of(state.rows.begin(),state.rows.end(),[](uint16_t row){return row==0;});
    const int event=spin_event(spin_kind,lines);
    const bool difficult=difficult_clear(lines,event);
    const bool was_active=state.b2b_active;
    const int chain=was_active ? state.b2b_chain : 0;
    int bonus=0,released=0,next_chain=chain;
    bool next_active=was_active;
    if (lines==0) {
        next_active=was_active; next_chain=chain;
    } else if (pc) {
        bonus=was_active?1:0; next_active=true; next_chain=was_active?chain+2:2;
    } else if (difficult) {
        bonus=was_active?1:0; next_active=true; next_chain=was_active?chain+1:0;
    } else {
        released=(was_active && chain>=4)?chain:0; next_active=false; next_chain=0;
    }
    int attack=base_attack(lines,event)+bonus+released;
    if (lines) {
        ++state.combo;
        if (state.combo>0) attack += std::min(4,state.combo/2+1);
    } else state.combo=-1;
    if (pc && lines) attack+=10;
    state.b2b_active=next_active;
    state.b2b_chain=static_cast<uint16_t>(std::max(0,next_chain));
    state.game_over=topped;
    for (int y=0;y<kHiddenRows;++y) if (state.rows[static_cast<size_t>(y)]!=0) state.game_over=true;
    const Features after=features(state.rows);
    value.lines=lines; value.attack=attack; value.spin_lines=event?lines:0; value.pc=pc; value.new_holes=std::max(0,after.holes-parent_holes);
    if (!state.game_over) {
        if (!spawn_next(state,queue)) state.game_over=true;
    }
    return true;
}

double event_score(const State& state,const LockValue& v) noexcept {
    return v.attack*0.85 + v.lines*0.760666 + v.spin_lines*1.25 + (v.pc?8.0:0.0) + v.new_holes*-1.2 + std::max<int>(0,state.combo)*0.03 + state.b2b_chain*0.05;
}

struct StateKey {
    std::array<uint16_t,kHeight> rows{};
    Piece current=Piece::None,hold=Piece::None;
    uint16_t queue_index=0;
    int16_t combo=-1;
    uint16_t b2b_chain=0;
    uint8_t flags=0;
    bool operator==(const StateKey& o) const noexcept {
        return rows==o.rows && current==o.current && hold==o.hold && queue_index==o.queue_index && combo==o.combo && b2b_chain==o.b2b_chain && flags==o.flags;
    }
};
struct StateHash {
    size_t operator()(const StateKey& s) const noexcept {
        uint64_t h=1469598103934665603ULL;
        auto mix=[&](uint64_t v){ h^=v; h*=1099511628211ULL; };
        for (uint16_t row:s.rows) mix(row);
        mix(static_cast<uint8_t>(s.current)); mix(static_cast<uint8_t>(s.hold)); mix(s.queue_index); mix(static_cast<uint16_t>(s.combo)); mix(s.b2b_chain); mix(s.flags);
        return static_cast<size_t>(h^(h>>32));
    }
};
StateKey key_of(const State& s) noexcept {
    StateKey k; k.rows=s.rows;k.current=s.current;k.hold=s.hold;k.queue_index=s.queue_index;k.combo=s.combo;k.b2b_chain=s.b2b_chain;
    k.flags=uint8_t(s.has_hold) | (uint8_t(s.can_hold)<<1) | (uint8_t(s.b2b_active)<<2) | (uint8_t(s.game_over)<<3);
    return k;
}

struct Node {
    State state{};
    double path_score=0.0;
    double rank_score=0.0;
    Move root{};
    bool has_root=false;
};

bool move_less(const Move& a,const Move& b) noexcept {
    if (a.use_hold!=b.use_hold) return a.use_hold<b.use_hold;
    if (a.piece!=b.piece) return static_cast<int>(a.piece)<static_cast<int>(b.piece);
    if (a.rotation!=b.rotation) return a.rotation<b.rotation;
    if (a.x!=b.x) return a.x<b.x;
    if (a.y!=b.y) return a.y<b.y;
    if (a.last_rotation!=b.last_rotation) return a.last_rotation<b.last_rotation;
    return a.kick_index<b.kick_index;
}
bool better_node(const Node& a,const Node& b) noexcept {
    if (a.rank_score!=b.rank_score) return a.rank_score>b.rank_score;
    if (a.path_score!=b.path_score) return a.path_score>b.path_score;
    if (a.state.game_over!=b.state.game_over) return !a.state.game_over;
    if (a.state.rows!=b.state.rows) return a.state.rows<b.state.rows;
    return move_less(a.root,b.root);
}

void insert_dedup(std::vector<Node>& children,std::unordered_map<StateKey,size_t,StateHash>& indices,Node&& child) {
    StateKey key=key_of(child.state);
    auto [it,inserted]=indices.emplace(std::move(key),children.size());
    if (inserted) children.push_back(std::move(child));
    else if (better_node(child,children[it->second])) children[it->second]=std::move(child);
}

void expand_branch(const Node& node,std::span<const Piece> queue,const Config& cfg,bool use_hold,int ply,std::vector<Node>& children,std::unordered_map<StateKey,size_t,StateHash>& indices) {
    State branch=node.state;
    Piece placing=branch.current;
    if (use_hold) {
        if (!branch.can_hold) return;
        const Piece outgoing=branch.current;
        if (branch.has_hold) placing=branch.hold;
        else {
            if (branch.queue_index>=queue.size()) return;
            placing=queue[branch.queue_index++];
            if (placing==Piece::None) return;
        }
        branch.hold=outgoing; branch.has_hold=true; branch.can_hold=false; branch.current=placing;
        if (collides(branch.rows,placing,3,1,0)) return;
    }
    const Features parent_features=features(branch.rows);
    const auto moves=reachable(branch.rows,placing,cfg.allow_180,cfg.max_nodes,use_hold);
    for (const Move& move:moves) {
        State child_state=branch;
        LockValue value;
        apply_placement(child_state,queue,move,parent_features.holes,value);
        const Features child_features=features(child_state.rows);
        Node child;
        child.state=child_state;
        child.path_score=node.path_score+event_score(child_state,value);
        child.rank_score=child.path_score+board_score(child_features,child_state.game_over);
        child.root = ply==0 ? move : node.root;
        child.has_root = true;
        insert_dedup(children,indices,std::move(child));
    }
}

} // namespace

Piece piece_from_char(char value) {
    switch(value){
        case 'I':case 'i':return Piece::I; case 'O':case 'o':return Piece::O; case 'T':case 't':return Piece::T;
        case 'S':case 's':return Piece::S; case 'Z':case 'z':return Piece::Z; case 'J':case 'j':return Piece::J; case 'L':case 'l':return Piece::L;
        default:return Piece::None;
    }
}
char piece_to_char(Piece piece) {
    constexpr std::array<char,7> names={'I','O','T','S','Z','J','L'};
    const int i=static_cast<int>(piece); return i>=0&&i<7?names[static_cast<size_t>(i)]:'?';
}

Result search(const State& root,std::span<const Piece> queue,const Config& raw) {
    Config cfg=raw; cfg.beam_width=std::max(1,cfg.beam_width); cfg.depth=std::max(1,cfg.depth); cfg.max_nodes=std::max(1,cfg.max_nodes);
    if (root.game_over || root.current==Piece::None) return {};
    Node start; start.state=root; const Features rootf=features(root.rows); start.rank_score=board_score(rootf,false);
    std::vector<Node> frontier{start};
    for (int ply=0;ply<cfg.depth;++ply) {
        std::vector<Node> children; children.reserve(static_cast<size_t>(cfg.beam_width)*48);
        std::unordered_map<StateKey,size_t,StateHash> indices; indices.reserve(static_cast<size_t>(cfg.beam_width)*64);
        for (const Node& node:frontier) {
            if (node.state.game_over) { if (node.has_root) insert_dedup(children,indices,Node(node)); continue; }
            expand_branch(node,queue,cfg,false,ply,children,indices);
            expand_branch(node,queue,cfg,true,ply,children,indices);
        }
        if (children.empty()) break;
        if (static_cast<int>(children.size())>cfg.beam_width) {
            auto middle=children.begin()+cfg.beam_width;
            std::nth_element(children.begin(),middle,children.end(),[](const Node&a,const Node&b){return better_node(a,b);});
            children.resize(static_cast<size_t>(cfg.beam_width));
        }
        std::sort(children.begin(),children.end(),[](const Node&a,const Node&b){return better_node(a,b);});
        frontier.swap(children);
    }
    if (frontier.empty()) return {};
    if (!frontier.front().has_root) return {};
    const Node& best = frontier.front();
    Result result; result.found=true; result.move=best.root; result.score=best.rank_score; return result;
}

} // namespace minoflux::oracle