#pragma once
// The data model. Everything here is designed around one question:
// what does the HOT LOOP (simulating one season, millions of times)
// actually need to touch? Answer: fixtures and probabilities. Team
// names, dates, actual results -- all of that is COLD data, needed only
// when loading input or writing output. Hot and cold are kept apart so
// the working set of the inner loop stays small and cache-friendly.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace simulator {

// Teams are integer ids (indices into SimulationInput::team_names), not
// objects holding strings. A Team class with a std::string name inside
// would drag heap pointers through the hot loop and make arrays of
// teams non-contiguous; an id is 2 bytes and indexes straight into
// flat arrays like points[20].
using TeamId = std::uint16_t;

enum class Outcome : std::uint8_t { HomeWin = 0, Draw = 1, AwayWin = 2 };

// One fixture, hot-loop edition: 12 bytes, trivially copyable, no
// pointers. The three outcome probabilities are stored PRE-ACCUMULATED
// as a CDF: cdf_home = P(H), cdf_draw = P(H) + P(D). Sampling an
// outcome from one uniform draw u is then two comparisons:
//     u < cdf_home  -> home win
//     u < cdf_draw  -> draw
//     otherwise     -> away win
// (P(A) never needs storing: it's the remaining probability mass.)
// float, not double: the probabilities carry ~2 significant digits of
// real information, so float's 7 are plenty, and halving the size keeps
// the whole 380-fixture season in ~4.5 KB -- a fraction of L1 cache.
struct Fixture {
    TeamId home;
    TeamId away;
    float cdf_home;
    float cdf_draw;
};

// Everything the engine needs, produced by the CSV loader. The struct
// is moved (not copied) from the loader to the caller; vectors move by
// swapping three pointers, so returning this by value is cheap.
struct SimulationInput {
    std::vector<std::string> team_names;    // index = TeamId (cold)
    std::vector<Fixture> fixtures;          // in date order (hot)
    std::vector<std::string> dates;         // per fixture (cold)
    std::vector<Outcome> actual_results;    // per fixture, for validation (cold)

    std::size_t team_count() const { return team_names.size(); }
};

}  // namespace simulator
