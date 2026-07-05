#pragma once
// Simulating ONE season: the hot loop. Everything in engine.hpp exists
// to call this function millions of times as fast as possible.
//
// simulate_season is a TEMPLATE over the random generator type. Why:
// the benchmark (bench/benchmark.cpp) races different generators
// through the identical loop, and the engine picks the winner -- with
// zero runtime cost. The alternatives are worse: a virtual
// RandomSource interface would add an indirect call per draw (~400
// per season) and block inlining; a function pointer likewise. A
// template parameter is compile-time polymorphism: each instantiation
// is generated with the generator's code inlined straight into the
// loop, as if hand-written for it. This is what "zero-overhead
// abstraction" means concretely.
//
// The constraint on URBG is the standard UniformRandomBitGenerator
// concept: result_type, min(), max(), operator()(). Anything
// satisfying it works -- std::mt19937_64, our Xoshiro256pp, and
// notably std::uniform_real_distribution and std::shuffle only require
// the same concept. (Being a template also forces the definition into
// this header: template bodies must be visible at the point of
// instantiation, or the linker hunts for symbols that were never
// generated -- the classic "undefined reference" template lesson.)

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "simulator/accumulator.hpp"
#include "simulator/types.hpp"

namespace simulator {

// Scratch buffers reused across seasons. Allocating these inside
// simulate_season would put two heap allocations (plus frees) in the
// hot loop; instead each worker owns one SeasonScratch for its entire
// run and every season just overwrites it. "Allocate once, reuse
// forever" is the single most effective micro-optimization pattern in
// simulation code.
struct SeasonScratch {
    std::vector<std::uint16_t> points;  // final points per team
    std::vector<TeamId> order;          // team at each rank after sorting

    explicit SeasonScratch(std::size_t n_teams)
        : points(n_teams, 0), order(n_teams, 0) {}
};

// Play out one season and record it into `acc`.
//
// RNG notes:
//  * A proper generator (mt19937_64, xoshiro256++), NOT rand(): rand()
//    has weak statistical quality, a tiny period, global hidden state
//    (a threading hazard), and rand() % n is biased.
//  * The generator is passed BY REFERENCE. Generators are stateful;
//    copying one would replay the same numbers.
//  * uniform_real_distribution maps raw 64-bit outputs to [0,1)
//    uniformly -- the correct way to get a probability draw.
template <class URBG>
void simulate_season(const SimulationInput& input,
                     SeasonScratch& scratch,
                     URBG& rng,
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
