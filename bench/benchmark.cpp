// Benchmark harness. Two experiments, output as markdown for the README:
//
//   1. RNG shootout: identical hot loop, mt19937_64 vs xoshiro256++.
//      This measures ONE variable -- generator cost -- because the
//      templated simulate_season guarantees everything else is the
//      same code.
//   2. Thread scaling: the full engine at 1..8 threads, with speedup
//      and parallel efficiency per row.
//
// Methodology (the part that makes numbers trustworthy):
//   * Release build assumed; benchmarking -O0 measures nothing.
//   * One untimed WARM-UP run first: it faults pages in, warms caches
//     and branch predictors, and lets the CPU leave idle power states,
//     so the first timed run isn't penalized for cold state.
//   * Each configuration runs 3 times; we report the MEDIAN -- robust
//     to a one-off hiccup (a background process stealing a core),
//     unlike the mean, and honest, unlike the min.
//   * A checksum from every run is accumulated and printed at the end,
//     so the compiler cannot dead-code-eliminate the very work being
//      timed, and identical-seed runs can be spot-checked for agreement.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "simulator/csv.hpp"
#include "simulator/engine.hpp"
#include "simulator/season.hpp"
#include "simulator/xoshiro.hpp"

namespace {

using simulator::Accumulator;
using simulator::SeasonScratch;
using simulator::SimulationInput;

// Construction differs per generator type (mt19937_64 wants a seed_seq,
// Xoshiro256pp takes (seed, stream) directly), so it's factored into a
// small factory template with explicit specializations -- the pattern
// for "same interface, type-specific setup".
template <class URBG>
URBG make_rng(std::uint64_t seed, std::uint64_t stream);

template <>
std::mt19937_64 make_rng(std::uint64_t seed, std::uint64_t stream) {
    std::seed_seq seq{seed, stream};
    return std::mt19937_64(seq);
}

template <>
simulator::Xoshiro256pp make_rng(std::uint64_t seed, std::uint64_t stream) {
    return simulator::Xoshiro256pp(seed, stream);
}

std::uint64_t g_checksum = 0;

std::uint64_t checksum(const Accumulator& acc) {
    std::uint64_t sum = acc.n_seasons();
    for (std::size_t t = 0; t < acc.n_teams(); ++t)
        for (std::size_t r = 0; r < acc.n_teams(); ++r)
            sum = sum * 31 + acc.rank_count(simulator::TeamId(t), r);
    return sum;
}

double median3(double a, double b, double c) {
    return std::max(std::min(a, b), std::min(std::max(a, b), c));
}

// Time one single-threaded run of `sims` seasons with generator URBG.
template <class URBG>
double time_hot_loop(const SimulationInput& input, std::uint64_t sims) {
    Accumulator acc(input.team_count());
    SeasonScratch scratch(input.team_count());
    URBG rng = make_rng<URBG>(42, 0);

    const auto t0 = std::chrono::steady_clock::now();
    for (std::uint64_t i = 0; i < sims; ++i)
        simulator::simulate_season(input, scratch, rng, acc);
    const auto t1 = std::chrono::steady_clock::now();

    g_checksum ^= checksum(acc);
    return std::chrono::duration<double>(t1 - t0).count();
}

template <class URBG>
double bench_rng(const SimulationInput& input, std::uint64_t sims) {
    time_hot_loop<URBG>(input, sims / 10);  // warm-up, untimed
    const double a = time_hot_loop<URBG>(input, sims);
    const double b = time_hot_loop<URBG>(input, sims);
    const double c = time_hot_loop<URBG>(input, sims);
    return double(sims) / median3(a, b, c);
}

double bench_engine(const SimulationInput& input, std::uint64_t sims,
                    unsigned threads) {
    const simulator::SimulationConfig config{sims, 42, threads};
    (void)simulator::run_simulations(input, {sims / 10, 42, threads});  // warm-up

    double times[3];
    for (double& t : times) {
        const auto t0 = std::chrono::steady_clock::now();
        const Accumulator acc = simulator::run_simulations(input, config);
        const auto t1 = std::chrono::steady_clock::now();
        t = std::chrono::duration<double>(t1 - t0).count();
        g_checksum ^= checksum(acc);
    }
    return double(sims) / median3(times[0], times[1], times[2]);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "data/fixtures_2025_26.csv";
    const SimulationInput input = simulator::load_fixtures_csv(path);
    const std::uint64_t sims = 1'000'000;

    std::printf("### RNG comparison (single thread, %llu seasons, median of 3)\n\n",
                (unsigned long long)sims);
    std::printf("| generator    | seasons/sec | relative |\n");
    std::printf("|--------------|-------------|----------|\n");
    const double mt = bench_rng<std::mt19937_64>(input, sims);
    std::printf("| mt19937_64   | %11.0f | 1.00x    |\n", mt);
    const double xo = bench_rng<simulator::Xoshiro256pp>(input, sims);
    std::printf("| xoshiro256++ | %11.0f | %.2fx    |\n\n", xo, xo / mt);

    std::printf("### Thread scaling (full engine, %llu seasons, median of 3)\n\n",
                (unsigned long long)sims);
    std::printf("| threads | seasons/sec | speedup | efficiency |\n");
    std::printf("|---------|-------------|---------|------------|\n");
    double base = 0.0;
    for (const unsigned t : {1u, 2u, 4u, 6u, 8u}) {
        const double rate = bench_engine(input, sims, t);
        if (t == 1) base = rate;
        std::printf("| %7u | %11.0f | %6.2fx | %9.0f%% |\n",
                    t, rate, rate / base, 100.0 * rate / base / t);
    }

    std::printf("\n(checksum %llx)\n", (unsigned long long)g_checksum);
    return 0;
}
