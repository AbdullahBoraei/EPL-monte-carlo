#include "simulator/engine.hpp"

#include <thread>
#include <vector>

#include "simulator/season.hpp"
#include "simulator/xoshiro.hpp"

namespace simulator {

unsigned resolve_thread_count(const SimulationConfig& config) {
    if (config.n_threads != 0) return config.n_threads;
    // hardware_concurrency() may return 0 if it can't tell; fall back to 1.
    const unsigned hw = std::thread::hardware_concurrency();
    return hw != 0 ? hw : 1;
}

Accumulator run_simulations(const SimulationInput& input,
                            const SimulationConfig& config) {
    const unsigned n_threads = resolve_thread_count(config);

    // Static work split: thread t runs a fixed share, with the first
    // `remainder` threads taking one extra so the counts sum exactly.
    // Static beats a work queue here because all seasons cost the same;
    // there is nothing for dynamic scheduling to balance, and a shared
    // queue would add synchronization for zero benefit.
    const std::uint64_t base = config.n_simulations / n_threads;
    const std::uint64_t remainder = config.n_simulations % n_threads;

    // One Accumulator per thread, allocated UP FRONT by the main thread.
    // Each worker writes only to partials[t] -- disjoint objects, and
    // each accumulator's count arrays live in their own heap blocks, so
    // threads don't share (or false-share) any cache lines they write.
    std::vector<Accumulator> partials;
    partials.reserve(n_threads);
    for (unsigned t = 0; t < n_threads; ++t)
        partials.emplace_back(input.team_count());

    const auto worker = [&input, &config](Accumulator& acc, unsigned t,
                                          std::uint64_t count) {
        // THE critical decision in parallel Monte Carlo: every thread
        // gets its own generator. Sharing one across threads would be a
        // data race (undefined behavior -- generator state updates are
        // not atomic); guarding it with a mutex would serialize the
        // very thing we parallelized. Seeding each stream with
        // (user seed, thread index) gives every thread a different,
        // well-mixed starting state, and keeps the whole run
        // reproducible: thread t always produces the same seasons.
        //
        // Generator choice is benchmark-driven, not guessed: the RNG
        // shootout in bench/benchmark.cpp measured xoshiro256++ at
        // ~1.8x mt19937_64 through this exact loop (RNG cost dominates
        // a 400-draws-per-season workload), with equal statistical
        // quality for this purpose. The engine uses the winner.
        Xoshiro256pp rng(config.seed, t);
        SeasonScratch scratch(acc.n_teams());

        for (std::uint64_t i = 0; i < count; ++i)
            simulate_season(input, scratch, rng, acc);
    };

    if (n_threads == 1) {
        // Run inline: measuring single-threaded throughput shouldn't
        // include thread startup, and debugging is nicer on one stack.
        worker(partials[0], 0, config.n_simulations);
        return std::move(partials[0]);
    }

    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (unsigned t = 0; t < n_threads; ++t)
        threads.emplace_back(worker, std::ref(partials[t]), t,
                             base + (t < remainder ? 1 : 0));
    // join() blocks until the thread finishes; joining ALL threads is
    // the synchronization point after which reading their accumulators
    // is safe (join formally "happens-after" everything the thread did).
    for (std::thread& th : threads) th.join();

    for (unsigned t = 1; t < n_threads; ++t) partials[0] += partials[t];
    return std::move(partials[0]);
}

}  // namespace simulator
