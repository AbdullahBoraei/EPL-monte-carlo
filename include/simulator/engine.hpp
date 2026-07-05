#pragma once
// The engine: run N seasons and return the aggregated statistics.
// Single-threaded for now; the next increment parallelizes this behind
// the same interface (callers won't change).

#include <cstdint>

#include "simulator/accumulator.hpp"
#include "simulator/types.hpp"

namespace simulator {

struct SimulationConfig {
    std::uint64_t n_simulations = 100'000;
    // A fixed seed makes every run reproducible: same seed, same input
    // -> bit-identical results. Essential for debugging ("that weird
    // table at sim #61,204") and for honest benchmarking (comparing
    // timings of identical work).
    std::uint64_t seed = 42;
};

Accumulator run_simulations(const SimulationInput& input,
                            const SimulationConfig& config);

}  // namespace simulator
