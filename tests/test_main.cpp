// Correctness tests for the simulation engine. No framework -- a CHECK
// macro and a process exit code are enough, and the repo stays
// dependency-free.
//
// The philosophy: a Monte Carlo engine's bugs don't crash, they QUIETLY
// SKEW DISTRIBUTIONS. So the tests verify against things that are known
// exactly: degenerate inputs with deterministic outcomes, analytically
// computable expectations, and structural invariants that must hold for
// every valid simulation regardless of randomness.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "simulator/engine.hpp"
#include "simulator/season.hpp"
#include "simulator/types.hpp"

static int failures = 0;

#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                 \
        }                                                               \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                 \
    do {                                                                      \
        const double a_ = (a), b_ = (b);                                      \
        if (std::abs(a_ - b_) > (tol)) {                                      \
            std::printf("FAIL %s:%d: %s = %.6f, expected %.6f (tol %.6f)\n",  \
                        __FILE__, __LINE__, #a, a_, b_, double(tol));         \
            ++failures;                                                       \
        }                                                                     \
    } while (0)

using namespace simulator;

namespace {

// Double round-robin fixture list for n teams with given probabilities
// per fixture (same probs everywhere unless overridden by the caller).
SimulationInput make_league(std::size_t n, float p_home, float p_draw) {
    SimulationInput input;
    for (std::size_t t = 0; t < n; ++t)
        input.team_names.push_back("Team" + std::to_string(t));
    for (std::size_t h = 0; h < n; ++h)
        for (std::size_t a = 0; a < n; ++a)
            if (h != a)
                input.fixtures.push_back(
                    Fixture{TeamId(h), TeamId(a), p_home, p_home + p_draw});
    return input;
}

}  // namespace

// A fully deterministic league: team i always beats team j when i < j.
// Every season must produce the identical table, so probabilities are
// exactly 0 or 1 and the points spread is exact.
void test_deterministic_league() {
    SimulationInput input = make_league(4, 0.0f, 0.0f);
    for (Fixture& fx : input.fixtures) {
        const bool home_stronger = fx.home < fx.away;
        fx.cdf_home = home_stronger ? 1.0f : 0.0f;
        fx.cdf_draw = fx.cdf_home;
    }

    const Accumulator acc = run_simulations(input, {.n_simulations = 1000, .seed = 1});

    // Team i wins against every weaker team, home and away:
    // team0: 6 wins = 18 pts, team1: 12, team2: 6, team3: 0.
    CHECK_NEAR(acc.mean_points(0), 18.0, 1e-12);
    CHECK_NEAR(acc.mean_points(1), 12.0, 1e-12);
    CHECK_NEAR(acc.mean_points(3), 0.0, 1e-12);
    CHECK_NEAR(acc.std_points(0), 0.0, 1e-9);          // no variance at all
    CHECK_NEAR(acc.rank_prob(0, 0), 1.0, 1e-12);       // always champion
    CHECK_NEAR(acc.rank_prob(3, 3), 1.0, 1e-12);       // always last
}

// Symmetric league where every match is a certain draw: all teams tie
// on points every season, so ranks are decided ONLY by the random
// tie-break -- which must be uniform (each team champion 1/n of the time).
void test_tiebreak_uniformity() {
    const std::size_t n = 4;
    const SimulationInput input = make_league(n, 0.0f, 1.0f);
    const std::uint64_t sims = 400'000;
    const Accumulator acc = run_simulations(input, {.n_simulations = sims, .seed = 2});

    // Binomial std for p=1/4 over 400k trials is ~0.0007; tolerance 4 sigma.
    for (std::size_t t = 0; t < n; ++t)
        CHECK_NEAR(acc.rank_prob(TeamId(t), 0), 0.25, 0.003);
}

// Expected points are analytically computable from the input
// probabilities: E[pts] = sum over matches of 3*P(win) + 1*P(draw).
// The simulated mean must converge to it (law of large numbers; the
// tolerance is a few standard errors of the mean).
void test_analytic_expected_points() {
    const std::size_t n = 6;
    SimulationInput input = make_league(n, 0.0f, 0.0f);
    // Varied, seeded-random probabilities -- not uniform ones, so an
    // indexing bug (home/away swap etc.) can't cancel out symmetrically.
    std::mt19937_64 gen(7);
    std::uniform_real_distribution<float> u(0.05f, 0.6f);
    for (Fixture& fx : input.fixtures) {
        const float ph = u(gen);
        const float pd = std::min(0.95f - ph, u(gen));
        fx.cdf_home = ph;
        fx.cdf_draw = ph + pd;
    }

    std::vector<double> expected(n, 0.0);
    for (const Fixture& fx : input.fixtures) {
        const double ph = fx.cdf_home, pd = fx.cdf_draw - fx.cdf_home;
        expected[fx.home] += 3.0 * ph + pd;
        expected[fx.away] += 3.0 * (1.0 - ph - pd) + pd;
    }

    const std::uint64_t sims = 200'000;
    const Accumulator acc = run_simulations(input, {.n_simulations = sims, .seed = 3});
    for (std::size_t t = 0; t < n; ++t)
        CHECK_NEAR(acc.mean_points(TeamId(t)), expected[t], 0.1);
}

// Structural invariants that hold for ANY input: every simulated season
// puts exactly one team at each rank, and each team at exactly one rank.
void test_histogram_invariants() {
    const std::size_t n = 5;
    const SimulationInput input = make_league(n, 0.4f, 0.3f);
    const std::uint64_t sims = 50'000;
    const Accumulator acc = run_simulations(input, {.n_simulations = sims, .seed = 4});

    for (std::size_t r = 0; r < n; ++r) {
        std::uint64_t total = 0;
        for (std::size_t t = 0; t < n; ++t) total += acc.rank_count(TeamId(t), r);
        CHECK(total == sims);  // exactly one team per rank per season
    }
    for (std::size_t t = 0; t < n; ++t) {
        std::uint64_t total = 0;
        for (std::size_t r = 0; r < n; ++r) total += acc.rank_count(TeamId(t), r);
        CHECK(total == sims);  // each team gets exactly one rank per season
    }
}

// Same seed -> bit-identical results; different seed -> different results.
void test_determinism() {
    const SimulationInput input = make_league(5, 0.4f, 0.3f);
    const Accumulator a = run_simulations(input, {.n_simulations = 20'000, .seed = 9});
    const Accumulator b = run_simulations(input, {.n_simulations = 20'000, .seed = 9});
    const Accumulator c = run_simulations(input, {.n_simulations = 20'000, .seed = 10});

    bool identical_ab = true, identical_ac = true;
    for (std::size_t t = 0; t < 5; ++t)
        for (std::size_t r = 0; r < 5; ++r) {
            identical_ab &= a.rank_count(TeamId(t), r) == b.rank_count(TeamId(t), r);
            identical_ac &= a.rank_count(TeamId(t), r) == c.rank_count(TeamId(t), r);
        }
    CHECK(identical_ab);
    CHECK(!identical_ac);
}

// Accumulator merge must be lossless: two half-runs merged equal the
// sum of their parts. (This is the property multithreading relies on.)
void test_accumulator_merge() {
    const SimulationInput input = make_league(5, 0.4f, 0.3f);
    Accumulator a = run_simulations(input, {.n_simulations = 10'000, .seed = 11});
    const Accumulator b = run_simulations(input, {.n_simulations = 15'000, .seed = 12});
    const std::uint64_t a_title = a.rank_count(0, 0), b_title = b.rank_count(0, 0);

    a += b;
    CHECK(a.n_seasons() == 25'000);
    CHECK(a.rank_count(0, 0) == a_title + b_title);
}

// Multithreaded runs must (1) simulate exactly the requested number of
// seasons even when it doesn't divide evenly by the thread count,
// (2) be bit-identical for the same seed and thread count, and
// (3) agree statistically with the single-threaded engine -- per-thread
// RNG streams change WHICH seasons are drawn, not their distribution.
void test_multithreaded() {
    const SimulationInput input = make_league(6, 0.4f, 0.3f);

    // 100'003 across 4 threads: the +1 remainder logic must make counts sum.
    const Accumulator a =
        run_simulations(input, {.n_simulations = 100'003, .seed = 5, .n_threads = 4});
    CHECK(a.n_seasons() == 100'003);

    const Accumulator b =
        run_simulations(input, {.n_simulations = 100'003, .seed = 5, .n_threads = 4});
    bool identical = true;
    for (std::size_t t = 0; t < 6; ++t)
        for (std::size_t r = 0; r < 6; ++r)
            identical &= a.rank_count(TeamId(t), r) == b.rank_count(TeamId(t), r);
    CHECK(identical);

    const Accumulator st =
        run_simulations(input, {.n_simulations = 200'000, .seed = 6, .n_threads = 1});
    const Accumulator mt =
        run_simulations(input, {.n_simulations = 200'000, .seed = 6, .n_threads = 8});
    for (std::size_t t = 0; t < 6; ++t) {
        CHECK_NEAR(mt.mean_points(TeamId(t)), st.mean_points(TeamId(t)), 0.15);
        CHECK_NEAR(mt.rank_prob(TeamId(t), 0), st.rank_prob(TeamId(t), 0), 0.01);
    }
}

// Forcing an outcome must make it happen in EVERY simulated season.
// In an otherwise all-draws 4-team league, forcing one home win gives
// that team 3+5*1 = 8 points (always champion) and the loser 5*1 = 5
// (always last); the untouched teams draw all 6 for 6 points each.
void test_force_outcome() {
    SimulationInput input = make_league(4, 0.0f, 1.0f);
    // fixtures are (h,a) pairs in loop order; fixture 0 is Team0 v Team1.
    force_outcome(input, 0, Outcome::HomeWin);

    const Accumulator acc = run_simulations(input, {.n_simulations = 20'000, .seed = 13});
    CHECK_NEAR(acc.mean_points(0), 8.0, 1e-12);
    CHECK_NEAR(acc.mean_points(1), 5.0, 1e-12);
    CHECK_NEAR(acc.mean_points(2), 6.0, 1e-12);
    CHECK_NEAR(acc.rank_prob(0, 0), 1.0, 1e-12);  // always champion
    CHECK_NEAR(acc.rank_prob(1, 3), 1.0, 1e-12);  // always last
}

int main() {
    test_deterministic_league();
    test_tiebreak_uniformity();
    test_analytic_expected_points();
    test_histogram_invariants();
    test_determinism();
    test_accumulator_merge();
    test_multithreaded();
    test_force_outcome();

    if (failures == 0) std::printf("all tests passed\n");
    else std::printf("%d check(s) FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
