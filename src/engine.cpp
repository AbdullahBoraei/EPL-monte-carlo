#include "simulator/engine.hpp"

#include <random>

#include "simulator/season.hpp"

namespace simulator {

Accumulator run_simulations(const SimulationInput& input,
                            const SimulationConfig& config) {
    Accumulator acc(input.team_count());
    SeasonScratch scratch(input.team_count());

    // seed_seq spreads a single seed across the Mersenne Twister's
    // 2.5 KB of internal state. Seeding mt19937_64 with a bare integer
    // (mt19937_64 rng{seed}) leaves most of that state poorly mixed for
    // the first few thousand draws; seed_seq is the <random>-idiomatic
    // way to initialize the full state properly.
    std::seed_seq seq{config.seed};
    std::mt19937_64 rng(seq);

    for (std::uint64_t i = 0; i < config.n_simulations; ++i)
        simulate_season(input, scratch, rng, acc);

    return acc;
}

}  // namespace simulator
