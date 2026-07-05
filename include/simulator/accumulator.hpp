#pragma once
// Statistics gathered across simulated seasons.
//
// The core data structure is the RANK HISTOGRAM: counts[team][rank] =
// how many simulated seasons ended with that team at that rank. Almost
// every question the engine answers falls out of it:
//     P(title)      = counts[t][0] / n
//     P(top 4)      = sum of counts[t][0..3] / n
//     P(relegation) = sum of counts[t][n-3..n-1] / n
// plus full finishing-position distributions for confidence intervals.
//
// Everything is stored as INTEGER counts and sums, not running
// averages: integers are exact (no float rounding drift over millions
// of seasons), and -- crucial for the threading increment -- two
// accumulators merge losslessly by simple addition. Each thread will
// fill its own Accumulator with no synchronization, and operator+=
// combines them at the end.

#include <cmath>
#include <cstdint>
#include <vector>

#include "simulator/types.hpp"

namespace simulator {

class Accumulator {
public:
    explicit Accumulator(std::size_t n_teams)
        : n_teams_(n_teams),
          rank_counts_(n_teams * n_teams, 0),
          points_sum_(n_teams, 0),
          points_sq_sum_(n_teams, 0) {}

    // Record one finished season. `order[r]` is the team at rank r
    // (0 = champion); `points[t]` is team t's final points tally.
    void record(const std::vector<TeamId>& order,
                const std::vector<std::uint16_t>& points) {
        ++n_seasons_;
        for (std::size_t rank = 0; rank < n_teams_; ++rank)
            ++rank_counts_[order[rank] * n_teams_ + rank];
        for (std::size_t t = 0; t < n_teams_; ++t) {
            points_sum_[t] += points[t];
            points_sq_sum_[t] += std::uint64_t(points[t]) * points[t];
        }
    }

    // Lossless merge -- the whole reason this class exists.
    Accumulator& operator+=(const Accumulator& other) {
        n_seasons_ += other.n_seasons_;
        for (std::size_t i = 0; i < rank_counts_.size(); ++i)
            rank_counts_[i] += other.rank_counts_[i];
        for (std::size_t t = 0; t < n_teams_; ++t) {
            points_sum_[t] += other.points_sum_[t];
            points_sq_sum_[t] += other.points_sq_sum_[t];
        }
        return *this;
    }

    std::uint64_t n_seasons() const { return n_seasons_; }
    std::size_t n_teams() const { return n_teams_; }

    std::uint64_t rank_count(TeamId t, std::size_t rank) const {
        return rank_counts_[t * n_teams_ + rank];
    }

    double rank_prob(TeamId t, std::size_t rank) const {
        return double(rank_count(t, rank)) / double(n_seasons_);
    }

    // P(team finishes at rank < `below`), e.g. below=4 -> top four.
    double prob_rank_below(TeamId t, std::size_t below) const {
        std::uint64_t c = 0;
        for (std::size_t r = 0; r < below; ++r) c += rank_count(t, r);
        return double(c) / double(n_seasons_);
    }

    double prob_rank_at_least(TeamId t, std::size_t from) const {
        std::uint64_t c = 0;
        for (std::size_t r = from; r < n_teams_; ++r) c += rank_count(t, r);
        return double(c) / double(n_seasons_);
    }

    double mean_points(TeamId t) const {
        return double(points_sum_[t]) / double(n_seasons_);
    }

    // Population standard deviation via E[X^2] - E[X]^2. The integer
    // sums make this exact up to the final division.
    double std_points(TeamId t) const {
        const double mean = mean_points(t);
        const double mean_sq = double(points_sq_sum_[t]) / double(n_seasons_);
        const double var = mean_sq - mean * mean;
        return var > 0.0 ? std::sqrt(var) : 0.0;  // clamp -1e-12 noise
    }

private:
    std::size_t n_teams_;
    std::uint64_t n_seasons_ = 0;
    std::vector<std::uint64_t> rank_counts_;   // [team * n_teams + rank]
    std::vector<std::uint64_t> points_sum_;    // per team
    std::vector<std::uint64_t> points_sq_sum_; // per team
};

}  // namespace simulator
