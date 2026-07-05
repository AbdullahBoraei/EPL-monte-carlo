#pragma once
// Simulating ONE season: the hot loop. Everything in engine.hpp exists
// to call this function millions of times as fast as possible.

#include <cstdint>
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
//  * std::mt19937_64 (Mersenne Twister), NOT rand(): rand() has weak
//    statistical quality, a tiny period, global hidden state (a
//    threading hazard), and rand() % n is biased. mt19937_64 has a
//    2^19937-1 period and passes standard statistical test suites.
//  * The engine passes the generator BY REFERENCE. Generators are
//    stateful; copying one would replay the same numbers.
//  * uniform_real_distribution maps raw 64-bit outputs to [0,1)
//    uniformly -- the correct way to get a probability draw.
void simulate_season(const SimulationInput& input,
                     SeasonScratch& scratch,
                     std::mt19937_64& rng,
                     Accumulator& acc);

}  // namespace simulator
