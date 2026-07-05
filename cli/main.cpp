// Thin CLI over the simulator library. All real logic lives in the
// library so it can be tested and reused; this file only parses
// arguments, calls it, and prints.

#include <cstdio>
#include <exception>

#include "simulator/csv.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <fixtures.csv>\n", argv[0]);
        return 2;
    }

    try {
        const simulator::SimulationInput input = simulator::load_fixtures_csv(argv[1]);

        std::printf("loaded %zu fixtures, %zu teams (%s .. %s)\n",
                    input.fixtures.size(), input.team_count(),
                    input.dates.front().c_str(), input.dates.back().c_str());

        // Aggregate the raw probabilities as a sanity check. Home
        // advantage should be visible: mean P(home win) > mean P(away win).
        double sum_home = 0.0, sum_draw = 0.0;
        for (const simulator::Fixture& fx : input.fixtures) {
            sum_home += fx.cdf_home;
            sum_draw += fx.cdf_draw - fx.cdf_home;
        }
        const double n = double(input.fixtures.size());
        std::printf("mean P(home)=%.3f  P(draw)=%.3f  P(away)=%.3f\n",
                    sum_home / n, sum_draw / n, 1.0 - (sum_home + sum_draw) / n);
        std::printf("sizeof(Fixture)=%zu bytes, season working set ~%zu bytes\n",
                    sizeof(simulator::Fixture),
                    sizeof(simulator::Fixture) * input.fixtures.size());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
