#pragma once
// Input boundary: parse the fixtures CSV exported by the Python model
// (scripts/export_fixtures.py in the EPL-match-prediction repo).
//
// Expected header:
//   match_id,date,home_team,away_team,p_home,p_draw,p_away,actual_result,fallback
//
// Errors (missing file, malformed rows, probabilities that don't sum
// to 1) throw std::runtime_error with the offending line number.
// Exceptions are the right tool HERE because this runs once at startup
// at I/O speed; the hot simulation loop never throws.

#include <string>

#include "simulator/accumulator.hpp"
#include "simulator/types.hpp"

namespace simulator {

SimulationInput load_fixtures_csv(const std::string& path);

// Output boundary: per-team probabilities and the full finishing-
// position distribution, one row per team, sorted by expected points.
// CSV so the Python side (pandas/matplotlib) ingests it in one line.
void write_results_csv(const std::string& path,
                       const SimulationInput& input,
                       const Accumulator& acc);

}  // namespace simulator
