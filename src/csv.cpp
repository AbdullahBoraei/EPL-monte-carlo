#include "simulator/csv.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace simulator {
namespace {

// Split one CSV line on commas. Our exporter never quotes fields (team
// names contain no commas), so a plain split is correct -- documented
// here so the simplification is a decision, not an oversight.
std::vector<std::string_view> split(std::string_view line) {
    std::vector<std::string_view> fields;
    std::size_t start = 0;
    while (true) {
        const std::size_t comma = line.find(',', start);
        if (comma == std::string_view::npos) {
            fields.push_back(line.substr(start));
            return fields;
        }
        fields.push_back(line.substr(start, comma - start));
        start = comma + 1;
    }
}

[[noreturn]] void fail(std::size_t line_no, const std::string& why) {
    throw std::runtime_error("fixtures CSV, line " + std::to_string(line_no) +
                             ": " + why);
}

float parse_prob(std::string_view s, std::size_t line_no) {
    // std::stof needs a std::string; fine at load time.
    float p = 0.0f;
    try {
        p = std::stof(std::string(s));
    } catch (const std::exception&) {
        fail(line_no, "not a number: '" + std::string(s) + "'");
    }
    if (p < 0.0f || p > 1.0f) fail(line_no, "probability out of [0,1]");
    return p;
}

Outcome parse_result(std::string_view s, std::size_t line_no) {
    if (s == "H") return Outcome::HomeWin;
    if (s == "D") return Outcome::Draw;
    if (s == "A") return Outcome::AwayWin;
    fail(line_no, "bad result: '" + std::string(s) + "'");
}

}  // namespace

SimulationInput load_fixtures_csv(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open fixtures CSV: " + path);

    SimulationInput input;
    // Team names are interned: first sighting assigns the next TeamId.
    // After loading, strings never appear again -- fixtures carry ids.
    std::unordered_map<std::string, TeamId> team_ids;
    const auto intern = [&](std::string_view name) -> TeamId {
        auto [it, inserted] =
            team_ids.try_emplace(std::string(name), TeamId(input.team_names.size()));
        if (inserted) input.team_names.emplace_back(name);
        return it->second;
    };

    std::string line;
    std::getline(file, line);  // header
    if (line.rfind("match_id,date,home_team,away_team,p_home", 0) != 0)
        throw std::runtime_error("unexpected CSV header -- wrong file? " + path);

    std::size_t line_no = 1;
    while (std::getline(file, line)) {
        ++line_no;
        if (line.empty()) continue;
        const auto f = split(line);
        if (f.size() != 9) fail(line_no, "expected 9 fields, got " + std::to_string(f.size()));

        const float p_home = parse_prob(f[4], line_no);
        const float p_draw = parse_prob(f[5], line_no);
        const float p_away = parse_prob(f[6], line_no);
        // The three probabilities must be a distribution. Tolerance is
        // loose (1e-4) because they crossed a text format; exactness is
        // restored below by deriving the CDF from p_home and p_draw only.
        if (std::abs((p_home + p_draw + p_away) - 1.0f) > 1e-4f)
            fail(line_no, "probabilities do not sum to 1");

        input.fixtures.push_back(Fixture{
            intern(f[2]), intern(f[3]),
            p_home, p_home + p_draw,
        });
        input.dates.emplace_back(f[1]);
        input.actual_results.push_back(parse_result(f[7], line_no));
    }

    if (input.fixtures.empty()) throw std::runtime_error("no fixtures in " + path);

    // Structural sanity: a round-robin season has every team playing
    // every other twice = 2*(n-1) fixtures per team.
    std::vector<int> played(input.team_count(), 0);
    for (const Fixture& fx : input.fixtures) {
        ++played[fx.home];
        ++played[fx.away];
    }
    const int expected = 2 * (int(input.team_count()) - 1);
    for (std::size_t t = 0; t < played.size(); ++t) {
        if (played[t] != expected)
            throw std::runtime_error(input.team_names[t] + " has " +
                                     std::to_string(played[t]) + " fixtures, expected " +
                                     std::to_string(expected));
    }

    return input;  // moved, not copied: vectors transfer by pointer swap
}

void write_results_csv(const std::string& path,
                       const SimulationInput& input,
                       const Accumulator& acc) {
    std::ofstream file(path);
    if (!file) throw std::runtime_error("cannot open for writing: " + path);

    const std::size_t n = acc.n_teams();

    // Sort teams by expected points for a readable file.
    std::vector<TeamId> teams(n);
    for (std::size_t t = 0; t < n; ++t) teams[t] = TeamId(t);
    std::sort(teams.begin(), teams.end(), [&acc](TeamId a, TeamId b) {
        return acc.mean_points(a) > acc.mean_points(b);
    });

    file << "team,expected_points,points_std,p_title,p_top4,p_relegation";
    for (std::size_t r = 0; r < n; ++r) file << ",p_rank_" << (r + 1);
    file << "\n";

    for (const TeamId t : teams) {
        file << input.team_names[t] << ',' << acc.mean_points(t) << ','
             << acc.std_points(t) << ',' << acc.rank_prob(t, 0) << ','
             << acc.prob_rank_below(t, 4) << ','
             << acc.prob_rank_at_least(t, n - 3);
        for (std::size_t r = 0; r < n; ++r) file << ',' << acc.rank_prob(t, r);
        file << "\n";
    }
}

}  // namespace simulator
