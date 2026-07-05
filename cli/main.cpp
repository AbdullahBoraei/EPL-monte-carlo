// Thin CLI over the simulator library. All real logic lives in the
// library so it can be tested and reused; this file only parses
// arguments, calls it, and prints.
//
// usage: simulate <fixtures.csv> [--sims N] [--seed S] [--out results.csv]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include "simulator/csv.hpp"
#include "simulator/engine.hpp"

namespace {

struct Args {
    std::string fixtures_path;
    std::string out_path;  // empty = don't write
    simulator::SimulationConfig config;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&]() -> std::string {
            if (++i >= argc) {
                std::fprintf(stderr, "missing value after %s\n", a.c_str());
                std::exit(2);
            }
            return argv[i];
        };
        if (a == "--sims") args.config.n_simulations = std::stoull(next());
        else if (a == "--seed") args.config.seed = std::stoull(next());
        else if (a == "--threads") args.config.n_threads = unsigned(std::stoul(next()));
        else if (a == "--out") args.out_path = next();
        else if (args.fixtures_path.empty()) args.fixtures_path = a;
        else { std::fprintf(stderr, "unexpected argument: %s\n", a.c_str()); std::exit(2); }
    }
    if (args.fixtures_path.empty()) {
        std::fprintf(stderr,
                     "usage: %s <fixtures.csv> [--sims N] [--seed S] "
                     "[--threads T] [--out results.csv]\n",
                     argv[0]);
        std::exit(2);
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    try {
        const simulator::SimulationInput input =
            simulator::load_fixtures_csv(args.fixtures_path);
        const std::size_t n = input.team_count();

        // The real final table (from actual_result), for side-by-side
        // comparison with what the simulation considered likely.
        std::vector<int> actual_points(n, 0);
        for (std::size_t i = 0; i < input.fixtures.size(); ++i) {
            const auto& fx = input.fixtures[i];
            switch (input.actual_results[i]) {
                case simulator::Outcome::HomeWin: actual_points[fx.home] += 3; break;
                case simulator::Outcome::Draw:    actual_points[fx.home] += 1;
                                                  actual_points[fx.away] += 1; break;
                case simulator::Outcome::AwayWin: actual_points[fx.away] += 3; break;
            }
        }

        // steady_clock for intervals: unlike system_clock it never jumps
        // (NTP adjustments, DST) -- the right clock for measuring durations.
        const auto t0 = std::chrono::steady_clock::now();
        const simulator::Accumulator acc = simulator::run_simulations(input, args.config);
        const auto t1 = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(t1 - t0).count();

        // Table sorted by expected points.
        std::vector<simulator::TeamId> teams(n);
        for (std::size_t t = 0; t < n; ++t) teams[t] = simulator::TeamId(t);
        std::sort(teams.begin(), teams.end(),
                  [&acc](simulator::TeamId a, simulator::TeamId b) {
                      return acc.mean_points(a) > acc.mean_points(b);
                  });

        std::printf("%-16s %11s %8s %7s %7s %10s\n",
                    "team", "xPts", "title", "top4", "releg", "actual pts");
        for (const simulator::TeamId t : teams) {
            std::printf("%-16s %5.1f %s %4.1f %6.1f%% %5.1f%% %6.1f%% %6d\n",
                        input.team_names[t].c_str(),
                        acc.mean_points(t), "\xC2\xB1", acc.std_points(t),
                        100.0 * acc.rank_prob(t, 0),
                        100.0 * acc.prob_rank_below(t, 4),
                        100.0 * acc.prob_rank_at_least(t, n - 3),
                        actual_points[t]);
        }

        std::printf("\n%llu seasons in %.3f s  (%.0f seasons/sec on %u thread%s)\n",
                    (unsigned long long)acc.n_seasons(), seconds,
                    double(acc.n_seasons()) / seconds,
                    simulator::resolve_thread_count(args.config),
                    simulator::resolve_thread_count(args.config) == 1 ? "" : "s");

        if (!args.out_path.empty()) {
            simulator::write_results_csv(args.out_path, input, acc);
            std::printf("wrote %s\n", args.out_path.c_str());
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
