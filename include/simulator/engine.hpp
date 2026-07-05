#pragma once
// The engine: run N seasons across worker threads and return the
// aggregated statistics.
//
// Parallelization strategy: Monte Carlo simulation is embarrassingly
// parallel -- seasons share no state, so the only design questions are
// (1) how threads get independent random numbers and (2) how results
// combine. Answers: a per-thread generator seeded from (seed, thread
// index), and per-thread Accumulators merged at the end. No mutex, no
// atomic, no shared mutable state exists anywhere in the hot path.

#include <cstdint>

#include "simulator/accumulator.hpp"
#include "simulator/types.hpp"

namespace simulator {

struct SimulationConfig {
    std::uint64_t n_simulations = 100'000;
    // A fixed seed makes runs reproducible: same seed + same input +
    // same thread count -> bit-identical results. Thread count matters
    // because each thread is its own RNG stream; scheduling does NOT
    // matter, because work is split statically, not stolen.
    std::uint64_t seed = 42;
    // 0 = one thread per hardware core (std::thread::hardware_concurrency).
    unsigned n_threads = 0;
};

Accumulator run_simulations(const SimulationInput& input,
                            const SimulationConfig& config);

// The thread count run_simulations will actually use for this config.
unsigned resolve_thread_count(const SimulationConfig& config);

}  // namespace simulator
