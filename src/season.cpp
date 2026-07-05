#include "simulator/season.hpp"

#include <algorithm>
#include <numeric>

namespace simulator {

void simulate_season(const SimulationInput& input,
                     SeasonScratch& scratch,
                     std::mt19937_64& rng,
                     Accumulator& acc) {
    auto& points = scratch.points;
    auto& order = scratch.order;
    std::fill(points.begin(), points.end(), std::uint16_t{0});

    // --- Play all fixtures -------------------------------------------
    // One uniform draw per match, classified against the fixture's
    // precomputed CDF. The distribution object is constructed once per
    // season, not per draw (it's cheap either way, but per-draw
    // construction is a habit worth not having).
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    for (const Fixture& fx : input.fixtures) {
        const float u = uniform(rng);
        if (u < fx.cdf_home) {
            points[fx.home] += 3;
        } else if (u < fx.cdf_draw) {
            points[fx.home] += 1;
            points[fx.away] += 1;
        } else {
            points[fx.away] += 3;
        }
    }

    // --- Build the final table ---------------------------------------
    // The model gives outcome probabilities, not scorelines, so goal
    // difference (the real EPL tiebreaker) is unmodeled. Points ties
    // are therefore broken UNIFORMLY AT RANDOM, which is symmetric: over
    // many seasons no team gains an artifact advantage. Implementation:
    // shuffle first, then STABLE sort by points. The stable sort
    // preserves the shuffled order within each points group -- i.e. a
    // uniformly random permutation of the tied teams. (A plain sort
    // would give an arbitrary-but-not-random, implementation-dependent
    // tie order; sorting 20 ids is trivial next to 380 RNG draws.)
    std::iota(order.begin(), order.end(), TeamId{0});
    std::shuffle(order.begin(), order.end(), rng);
    std::stable_sort(order.begin(), order.end(),
                     [&points](TeamId a, TeamId b) { return points[a] > points[b]; });

    acc.record(order, points);
}

}  // namespace simulator
