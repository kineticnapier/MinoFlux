#include "oracle_core.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace minoflux::oracle {
namespace {

struct Cell { int8_t x; int8_t y; };
constexpr Cell C(int x,int y){ return Cell{static_cast<int8_t>(x),static_cast<int8_t>(y)}; }
using Shape=std::array<Cell,4>;
using Rotations=std::array<Shape,4>;
constexpr std::array<Rotations,7> kShapes={
    Rotations{Shape{C(0,1),C(1,1),C(2,1),C(3,1)},Shape{C(2,0),C(2,1),C(2,2),C(2,3)},Shape{C(0,2),C(1,2),C(2,2),C(3,2)},Shape{C(1,0),C(1,1),C(1,2),C(1,3)}},
    Rotations{Shape{C(1,0),C(2,0),C(1,1),C(2,1)},Shape{C(1,0),C(2,0),C(1,1),C(2,1)},Shape{C(1,0),C(2,0),C(1,1),C(2,1)},Shape{C(1,0),C(2,0),C(1,1),C(2,1)}},
    Rotations{Shape{C(1,0),C(0,1),C(1,1),C(2,1)},Shape{C(1,0),C(1,1),C(2,1),C(1,2)},Shape{C(0,1),C(1,1),C(2,1),C(1,2)},Shape{C(1,0),C(0,1),C(1,1),C(1,2)}},
    Rotations{Shape{C(1,0),C(2,0),C(0,1),C(1,1)},Shape{C(1,0),C(1,1),C(2,1),C(2,2)},Shape{C(1,1),C(2,1),C(0,2),C(1,2)},Shape{C(0,0),C(0,1),C(1,1),C(1,2)}},
    Rotations{Shape{C(0,0),C(1,0),C(1,1),C(2,1)},Shape{C(2,0),C(1,1),C(2,1),C(1,2)},Shape{C(0,1),C(1,1),C(1,2),C(2,2)},Shape{C(1,0),C(0,1),C(1,1),C(0,2)}},
    Rotations{Shape{C(0,0),C(0,1),C(1,1),C(2,1)},Shape{C(1,0),C(2,0),C(1,1),C(1,2)},Shape{C(0,1),C(1,1),C(2,1),C(2,2)},Shape{C(1,0),C(1,1),C(0,2),C(1,2)}},
    Rotations{Shape{C(2,0),C(0,1),C(1,1),C(2,1)},Shape{C(1,0),C(1,1),C(1,2),C(2,2)},Shape{C(0,1),C(1,1),C(2,1),C(0,2)},Shape{C(0,0),C(1,0),C(1,1),C(1,2)}}
};

bool collides(const std::array<uint16_t,kHeight>& rows,Piece piece,int x,int y,int rotation) noexcept {
    const auto& cells=kShapes[static_cast<size_t>(piece)][static_cast<size_t>(rotation&3)];
    for(const Cell c:cells){ const int cx=x+c.x,cy=y+c.y; if(cx<0||cx>=kWidth||cy>=kHeight)return true; if(cy>=0&&(rows[static_cast<size_t>(cy)]&(uint16_t{1}<<cx)))return true; }
    return false;
}
bool occupied_or_wall(const std::array<uint16_t,kHeight>& rows,int x,int y) noexcept {
    if(x<0||x>=kWidth||y<0||y>=kHeight)return true;
    return (rows[static_cast<size_t>(y)]&(uint16_t{1}<<x))!=0;
}
int classify_t_spin(const std::array<uint16_t,kHeight>& rows,int x,int y,int rotation,int kick_index) noexcept {
    const int px=x+1,py=y+1;
    const std::array<bool,4> corners={occupied_or_wall(rows,px-1,py-1),occupied_or_wall(rows,px+1,py-1),occupied_or_wall(rows,px-1,py+1),occupied_or_wall(rows,px+1,py+1)};
    if(int(corners[0])+int(corners[1])+int(corners[2])+int(corners[3])<3)return 0;
    constexpr std::array<std::array<int,2>,4> front={{{{0,1}},{{1,3}},{{2,3}},{{0,2}}}};
    const auto p=front[static_cast<size_t>(rotation&3)];
    return (corners[p[0]]&&corners[p[1]])||kick_index==4 ? 2 : 1;
}

struct Features{int aggregate_height=0,max_height=0,holes=0,hole_depth=0,bumpiness=0,wells=0,t_spin_slots=0;};
bool cell_occupied(const std::array<uint16_t,kHeight>& rows,int x,int y) noexcept { if(x<0||x>=kWidth||y<0||y>=kHeight)return true; return (rows[static_cast<size_t>(y)]&(uint16_t{1}<<x))!=0; }
bool cell_empty(const std::array<uint16_t,kHeight>& rows,int x,int y) noexcept { return x>=0&&x<kWidth&&y>=0&&y<kHeight&&!cell_occupied(rows,x,y); }
Features features(const std::array<uint16_t,kHeight>& rows){
    Features f; std::array<int,kWidth> heights{};
    for(int x=0;x<kWidth;++x){int top=-1,occupied_above=0;for(int y=0;y<kHeight;++y){const bool occ=(rows[static_cast<size_t>(y)]&(uint16_t{1}<<x))!=0;if(occ){if(top<0)top=y;++occupied_above;}else if(top>=0){++f.holes;f.hole_depth+=occupied_above;}}heights[static_cast<size_t>(x)]=top<0?0:kHeight-top;}
    for (int h : heights) { f.aggregate_height += h; }
    f.max_height=*std::max_element(heights.begin(),heights.end());
    for(int x=0;x<kWidth-1;++x)f.bumpiness+=std::abs(heights[static_cast<size_t>(x)]-heights[static_cast<size_t>(x+1)]);
    for(int x=0;x<kWidth;++x){int run=0;for(int y=0;y<kHeight;++y){const bool well=!cell_occupied(rows,x,y)&&(x==0||cell_occupied(rows,x-1,y))&&(x==kWidth-1||cell_occupied(rows,x+1,y));if(well)++run;else{f.wells+=run*(run+1)/2;run=0;}}f.wells+=run*(run+1)/2;}
    for(int y=0;y<kHeight;++y)for(int x=0;x<kWidth;++x){if(cell_occupied(rows,x,y))continue;const int corners=int(cell_occupied(rows,x-1,y-1))+int(cell_occupied(rows,x+1,y-1))+int(cell_occupied(rows,x-1,y+1))+int(cell_occupied(rows,x+1,y+1));if(corners<3)continue;const int cardinals=int(cell_empty(rows,x,y-1))+int(cell_empty(rows,x,y+1))+int(cell_empty(rows,x-1,y))+int(cell_empty(rows,x+1,y));if(cardinals>=3)++f.t_spin_slots;}
    return f;
}
double board_score(const Features& f,bool game_over) noexcept { const double density=double(f.t_spin_slots)/(1.0+double(f.holes)); double score=f.aggregate_height*-0.510066+f.max_height*-0.08+f.holes*-0.8+f.hole_depth*-0.12+f.bumpiness*-0.184483+f.wells*-0.06+f.t_spin_slots*1.2+density*0.28;if(game_over)score-=1'000'000.0;return score; }
int spin_event(int kind,int lines) noexcept { if(!kind)return 0;if(kind==1){if(lines==0)return 1;if(lines==1)return 2;return lines==2?5:6;}if(lines==0)return 3;if(lines==1)return 4;if(lines==2)return 5;return 6; }
int base_attack(int lines,int event) noexcept { if(event==2)return 0;if(event==4)return 2;if(event==5)return 4;if(event==6)return 6;switch(lines){case 2:return 1;case 3:return 2;case 4:return 4;default:return 0;} }
bool difficult_clear(int lines,int event) noexcept { return lines==4||(event!=0&&lines>0); }
struct LockValue{int lines=0,attack=0,spin_lines=0;bool pc=false;int new_holes=0;};
bool spawn_next(State& state,std::span<const Piece> queue) noexcept { if(state.queue_index>=queue.size())return false;state.current=queue[state.queue_index++];state.can_hold=true;return state.current!=Piece::None&&!collides(state.rows,state.current,3,1,0); }
bool apply_placement(State& state,std::span<const Piece> queue,const Move& move,int parent_holes,LockValue& value){
    const int spin_kind=(move.piece==Piece::T&&move.last_rotation)?classify_t_spin(state.rows,move.x,move.y,move.rotation,move.kick_index):0;bool topped=false;
    const auto& cells=kShapes[static_cast<size_t>(move.piece)][static_cast<size_t>(move.rotation&3)];
    for(const Cell c:cells){const int x=move.x+c.x,y=move.y+c.y;if(y<0)topped=true;else state.rows[static_cast<size_t>(y)]|=uint16_t{1}<<x;}
    std::array<uint16_t,kHeight> cleared{};int write=kHeight-1,lines=0;for(int y=kHeight-1;y>=0;--y){if(state.rows[static_cast<size_t>(y)]==kFullRow)++lines;else cleared[static_cast<size_t>(write--)]=state.rows[static_cast<size_t>(y)];}state.rows=cleared;
    const bool pc=std::all_of(state.rows.begin(),state.rows.end(),[](uint16_t row){return row==0;});const int event=spin_event(spin_kind,lines);const bool difficult=difficult_clear(lines,event);const bool was_active=state.b2b_active;const int chain=was_active?state.b2b_chain:0;int bonus=0,released=0,next_chain=chain;bool next_active=was_active;
    if(lines==0){}else if(pc){bonus=was_active?1:0;next_active=true;next_chain=was_active?chain+2:2;}else if(difficult){bonus=was_active?1:0;next_active=true;next_chain=was_active?chain+1:0;}else{released=(was_active&&chain>=4)?chain:0;next_active=false;next_chain=0;}
    int attack=base_attack(lines,event)+bonus+released;if(lines){++state.combo;if(state.combo>0)attack+=std::min(4,state.combo/2+1);}else state.combo=-1;if(pc&&lines)attack+=10;state.b2b_active=next_active;state.b2b_chain=static_cast<uint16_t>(std::max(0,next_chain));state.game_over=topped;for(int y=0;y<kHiddenRows;++y)if(state.rows[static_cast<size_t>(y)]!=0)state.game_over=true;
    const Features after=features(state.rows);value.lines=lines;value.attack=attack;value.spin_lines=event?lines:0;value.pc=pc;value.new_holes=std::max(0,after.holes-parent_holes);if(!state.game_over&&!spawn_next(state,queue))state.game_over=true;return true;
}
double event_score(const State& state,const LockValue& v) noexcept { return v.attack*0.85+v.lines*0.760666+v.spin_lines*1.25+(v.pc?8.0:0.0)+v.new_holes*-1.2+std::max<int>(0,state.combo)*0.03+state.b2b_chain*0.05; }

struct StateKey{std::array<uint16_t,kHeight> rows{};Piece current=Piece::None,hold=Piece::None;uint16_t queue_index=0;int16_t combo=-1;uint16_t b2b_chain=0;uint8_t flags=0;bool operator==(const StateKey& o)const noexcept{return rows==o.rows&&current==o.current&&hold==o.hold&&queue_index==o.queue_index&&combo==o.combo&&b2b_chain==o.b2b_chain&&flags==o.flags;}};
struct StateHash{size_t operator()(const StateKey& s)const noexcept{uint64_t h=1469598103934665603ULL;auto mix=[&](uint64_t v){h^=v;h*=1099511628211ULL;};for(uint16_t row:s.rows)mix(row);mix(static_cast<uint8_t>(s.current));mix(static_cast<uint8_t>(s.hold));mix(s.queue_index);mix(static_cast<uint16_t>(s.combo));mix(s.b2b_chain);mix(s.flags);return static_cast<size_t>(h^(h>>32));}};
StateKey key_of(const State& s) noexcept { StateKey k;k.rows=s.rows;k.current=s.current;k.hold=s.hold;k.queue_index=s.queue_index;k.combo=s.combo;k.b2b_chain=s.b2b_chain;k.flags=uint8_t(s.has_hold)|(uint8_t(s.can_hold)<<1)|(uint8_t(s.b2b_active)<<2)|(uint8_t(s.game_over)<<3);return k; }
struct Node{State state{};double path_score=0.0,rank_score=0.0;Move root{};bool has_root=false;};
bool move_less(const Move&a,const Move&b) noexcept {if(a.use_hold!=b.use_hold)return a.use_hold<b.use_hold;if(a.piece!=b.piece)return static_cast<int>(a.piece)<static_cast<int>(b.piece);if(a.rotation!=b.rotation)return a.rotation<b.rotation;if(a.x!=b.x)return a.x<b.x;if(a.y!=b.y)return a.y<b.y;if(a.last_rotation!=b.last_rotation)return a.last_rotation<b.last_rotation;return a.kick_index<b.kick_index;}
bool better_node(const Node&a,const Node&b) noexcept {if(a.rank_score!=b.rank_score)return a.rank_score>b.rank_score;if(a.path_score!=b.path_score)return a.path_score>b.path_score;if(a.state.game_over!=b.state.game_over)return !a.state.game_over;if(a.state.rows!=b.state.rows)return a.state.rows<b.state.rows;return move_less(a.root,b.root);}
void insert_dedup(std::vector<Node>&children,std::unordered_map<StateKey,size_t,StateHash>&indices,Node&&child){StateKey key=key_of(child.state);auto[it,inserted]=indices.emplace(std::move(key),children.size());if(inserted)children.push_back(std::move(child));else if(better_node(child,children[it->second]))children[it->second]=std::move(child);}
void expand_branch(const Node&node,std::span<const Piece>queue,const Config&cfg,bool use_hold,int ply,std::vector<Node>&children,std::unordered_map<StateKey,size_t,StateHash>&indices){
    State branch=node.state;Piece placing=branch.current;if(use_hold){if(!branch.can_hold)return;const Piece outgoing=branch.current;if(branch.has_hold)placing=branch.hold;else{if(branch.queue_index>=queue.size())return;placing=queue[branch.queue_index++];if(placing==Piece::None)return;}branch.hold=outgoing;branch.has_hold=true;branch.can_hold=false;branch.current=placing;if(collides(branch.rows,placing,3,1,0))return;}
    const Features parent_features=features(branch.rows);const auto moves=reachable_moves(branch.rows,placing,cfg.allow_180,cfg.max_nodes,use_hold);for(const Move&move:moves){State child_state=branch;LockValue value;apply_placement(child_state,queue,move,parent_features.holes,value);const Features child_features=features(child_state.rows);Node child;child.state=child_state;child.path_score=node.path_score+event_score(child_state,value);child.rank_score=child.path_score+board_score(child_features,child_state.game_over);child.root=ply==0?move:node.root;child.has_root=true;insert_dedup(children,indices,std::move(child));}
}
} // namespace

Piece piece_from_char(char value){switch(value){case'I':case'i':return Piece::I;case'O':case'o':return Piece::O;case'T':case't':return Piece::T;case'S':case's':return Piece::S;case'Z':case'z':return Piece::Z;case'J':case'j':return Piece::J;case'L':case'l':return Piece::L;default:return Piece::None;}}
char piece_to_char(Piece piece){constexpr std::array<char,7>names={'I','O','T','S','Z','J','L'};const int i=static_cast<int>(piece);return i>=0&&i<7?names[static_cast<size_t>(i)]:'?';}
Result search(const State&root,std::span<const Piece>queue,const Config&raw){Config cfg=raw;cfg.beam_width=std::max(1,cfg.beam_width);cfg.depth=std::max(1,cfg.depth);cfg.max_nodes=std::max(1,cfg.max_nodes);if(root.game_over||root.current==Piece::None)return{};Node start;start.state=root;start.rank_score=board_score(features(root.rows),false);std::vector<Node>frontier{start};for(int ply=0;ply<cfg.depth;++ply){std::vector<Node>children;children.reserve(static_cast<size_t>(cfg.beam_width)*48);std::unordered_map<StateKey,size_t,StateHash>indices;indices.reserve(static_cast<size_t>(cfg.beam_width)*64);for(const Node&node:frontier){if(node.state.game_over){if(node.has_root)insert_dedup(children,indices,Node(node));continue;}expand_branch(node,queue,cfg,false,ply,children,indices);expand_branch(node,queue,cfg,true,ply,children,indices);}if(children.empty())break;if(static_cast<int>(children.size())>cfg.beam_width){auto middle=children.begin()+cfg.beam_width;std::nth_element(children.begin(),middle,children.end(),[](const Node&a,const Node&b){return better_node(a,b);});children.resize(static_cast<size_t>(cfg.beam_width));}std::sort(children.begin(),children.end(),[](const Node&a,const Node&b){return better_node(a,b);});frontier.swap(children);}if(frontier.empty()||!frontier.front().has_root)return{};const Node&best=frontier.front();Result result;result.found=true;result.move=best.root;result.score=best.rank_score;return result;}

} // namespace minoflux::oracle
