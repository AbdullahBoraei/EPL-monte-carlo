#pragma once
// xoshiro256++ 1.0 -- David Blackman & Sebastiano Vigna's generator
// (public-domain reference implementation, prng.di.unimi.it).
//
// Why offer an alternative to std::mt19937_64 at all? The Mersenne
// Twister is statistically excellent but heavyweight: 2.5 KB of state
// and a relatively expensive update. In a Monte Carlo hot loop where
// ~400 draws happen per season, generator cost is a first-order term.
// xoshiro256++ keeps 32 BYTES of state (fits in two SIMD registers),
// needs a handful of xor/shift/rotate ops per draw, and still passes
// the stringent statistical test suites (BigCrush, PractRand). It is
// what NumPy-adjacent scientific code reaches for; the benchmark in
// bench/ measures what it buys us here.
//
// The class satisfies the standard UniformRandomBitGenerator concept
// (result_type / min / max / operator()), so it drops into
// uniform_real_distribution, std::shuffle, and our templated
// simulate_season exactly like a standard generator.

#include <cstdint>

namespace simulator {

class Xoshiro256pp {
public:
    using result_type = std::uint64_t;
    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return ~result_type{0}; }

    // (seed, stream) mirrors the engine's per-thread scheme: same seed +
    // different stream -> a completely different, decorrelated sequence.
    // State is expanded from the two values via splitmix64, the seeding
    // recipe the xoshiro authors specify: it guarantees a well-mixed,
    // never-all-zero state even from small seeds like 0 or 1.
    explicit Xoshiro256pp(std::uint64_t seed, std::uint64_t stream = 0) {
        std::uint64_t x = seed ^ (0x9e3779b97f4a7c15ULL * (stream + 1));
        for (auto& word : state_) word = splitmix64(x);
    }

    result_type operator()() {
        const std::uint64_t result = rotl(state_[0] + state_[3], 23) + state_[0];
        const std::uint64_t t = state_[1] << 17;
        state_[2] ^= state_[0];
        state_[3] ^= state_[1];
        state_[1] ^= state_[2];
        state_[0] ^= state_[3];
        state_[2] ^= t;
        state_[3] = rotl(state_[3], 45);
        return result;
    }

private:
    static std::uint64_t rotl(std::uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    static std::uint64_t splitmix64(std::uint64_t& x) {
        std::uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    std::uint64_t state_[4];
};

}  // namespace simulator
