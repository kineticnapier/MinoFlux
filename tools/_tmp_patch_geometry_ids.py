from pathlib import Path

path = Path('src/minoflux_ai/_reachability_native.cpp')
text = path.read_text()


def replace_once(old: str, new: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f'expected one match, got {count}: {old[:120]!r}')
    text = text.replace(old, new, 1)


replace_once(
'''    std::vector<Mask256> collision_masks;
    std::vector<Mask256> geometry_masks;

    std::vector<uint32_t> state_group_offsets;
''',
'''    std::vector<Mask256> collision_masks;
    std::vector<Mask256> geometry_masks;
    std::vector<int32_t> geometry_ids;
    int32_t geometry_count = 0;

    std::vector<uint32_t> state_group_offsets;
''')

replace_once(
'''    std::vector<int32_t> visited_rotation_ids;
    std::vector<int32_t> landing_trail;
};
''',
'''    std::vector<int32_t> visited_rotation_ids;
    std::vector<int32_t> landing_trail;
    std::vector<BestRecord> best_records;
    std::vector<uint32_t> best_generations;
    std::vector<int32_t> touched_geometry_ids;
    uint32_t best_generation = 0;
};
''')

replace_once(
'''    std::unordered_map<Mask256, BestRecord, MaskHash> best;
    best.reserve(128);
    const auto representative_started = Clock::now();
''',
'''    const size_t geometry_count = static_cast<size_t>(table.geometry_count);
    if (scratch.best_records.size() < geometry_count) {
        scratch.best_records.resize(geometry_count);
    }
    if (scratch.best_generations.size() < geometry_count) {
        scratch.best_generations.resize(geometry_count, 0);
    }
    scratch.touched_geometry_ids.clear();
    scratch.touched_geometry_ids.reserve(geometry_count);
    ++scratch.best_generation;
    if (scratch.best_generation == 0) {
        std::fill(scratch.best_generations.begin(), scratch.best_generations.end(), uint32_t{0});
        scratch.best_generation = 1;
    }
    const uint32_t best_generation = scratch.best_generation;
    const auto representative_started = Clock::now();
''')

replace_once(
'''            auto it = best.find(key);
            if (it == best.end()) {
                best.emplace(key, candidate);
            } else if (better_non_t(candidate, it->second)) {
                it->second = candidate;
            }
''',
'''            const int32_t geometry_id = table.geometry_ids[static_cast<size_t>(final_state)];
            auto& generation = scratch.best_generations[static_cast<size_t>(geometry_id)];
            BestRecord& current = scratch.best_records[static_cast<size_t>(geometry_id)];
            if (generation != best_generation) {
                generation = best_generation;
                scratch.touched_geometry_ids.push_back(geometry_id);
                current = candidate;
            } else if (better_non_t(candidate, current)) {
                current = candidate;
            }
''')

replace_once(
'''                auto it = best.find(key);
                if (it == best.end()) {
                    best.emplace(key, candidate);
                } else if (better_t(candidate, it->second)) {
                    it->second = candidate;
                }
''',
'''                const int32_t geometry_id = table.geometry_ids[static_cast<size_t>(final_state)];
                auto& generation = scratch.best_generations[static_cast<size_t>(geometry_id)];
                BestRecord& current = scratch.best_records[static_cast<size_t>(geometry_id)];
                if (generation != best_generation) {
                    generation = best_generation;
                    scratch.touched_geometry_ids.push_back(geometry_id);
                    current = candidate;
                } else if (better_t(candidate, current)) {
                    current = candidate;
                }
''')

# The geometry mask itself is no longer hashed in the hot loop.
text = text.replace(
'''            const Mask256& key = table.geometry_masks[static_cast<size_t>(final_state)];
''',
'',
1,
)
text = text.replace(
'''                const Mask256& key = table.geometry_masks[static_cast<size_t>(final_state)];
''',
'',
1,
)

replace_once(
'''    result.placements.reserve(best.size());
    for (const auto& entry : best) {
        result.placements.push_back(entry.second.placement);
    }
''',
'''    result.placements.reserve(scratch.touched_geometry_ids.size());
    for (int32_t geometry_id : scratch.touched_geometry_ids) {
        result.placements.push_back(
            scratch.best_records[static_cast<size_t>(geometry_id)].placement
        );
    }
''')

replace_once(
'''    table->collision_masks = decode_masks(collision_masks, state_count, "collision_masks");
    table->geometry_masks = decode_masks(geometry_masks, state_count, "geometry_masks");

    table->state_group_offsets.reserve(state_count + 1);
''',
'''    table->collision_masks = decode_masks(collision_masks, state_count, "collision_masks");
    table->geometry_masks = decode_masks(geometry_masks, state_count, "geometry_masks");
    table->geometry_ids.assign(state_count, -1);
    std::unordered_map<Mask256, int32_t, MaskHash> geometry_lookup;
    geometry_lookup.reserve(state_count);
    for (size_t state_id = 0; state_id < state_count; ++state_id) {
        if (table->geometry_invalid[state_id] != 0) {
            continue;
        }
        const Mask256& mask = table->geometry_masks[state_id];
        const auto inserted = geometry_lookup.emplace(mask, table->geometry_count);
        if (inserted.second) {
            ++table->geometry_count;
        }
        table->geometry_ids[state_id] = inserted.first->second;
    }

    table->state_group_offsets.reserve(state_count + 1);
''')

path.write_text(text)
print('patched geometry representative IDs')
