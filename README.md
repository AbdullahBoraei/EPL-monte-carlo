# epl-monte-carlo

A multithreaded Monte Carlo season simulator in modern C++ (C++20, zero
dependencies). It takes per-match outcome probabilities produced by my
[EPL match predictor](https://github.com/AbdullahBoraei/EPL-match-prediction)
(Python, logistic regression) and plays out the full Premier League season
**over a million times per second** to answer questions a single prediction
can't:

- What is each team's probability of winning the title, making the top 4, or
  being relegated?
- What does the expected final table look like — with uncertainty, not just
  point estimates?
- How much does one specific result swing the title race? (what-if analysis)

## The two-language pipeline

```
EPL-match-prediction (Python)                epl-monte-carlo (C++)
┌─────────────────────────────┐   CSV    ┌──────────────────────────────┐
│ historical data → features  │ ───────> │ 380 fixtures × P(H/D/A)      │
│ → logistic regression       │ fixtures │ → simulate season ×10M       │
│ → P(H/D/A) per fixture      │          │ → rank histogram per team    │
└─────────────────────────────┘          └──────────────────────────────┘
                                                      │ CSV
                                                      v
                                          results: P(title), P(top4),
                                          P(relegation), xPts ± σ
```

Each language does what it's best at: Python owns modeling (pandas,
scikit-learn); C++ owns the compute-bound simulation. The interface is
deliberately boring — a CSV each way — so either side can be swapped out.
`scripts/export_fixtures.py` in the predictor repo produces the input;
a copy for 2025-26 ships in `data/` so this repo runs standalone.

## Quick start

```bash
cmake -B build -S .        # Release (-O3) by default
cmake --build build
./build/tests              # correctness suite
./build/simulate data/fixtures_2025_26.csv --sims 1000000 --out results.csv
```

Sample output (1M simulated 2025-26 seasons, model probabilities fixed
before the season):

```
team                    xPts    title    top4   releg actual pts
Arsenal           73.3 ±  7.5   38.6%  87.4%    0.0%     85
Man City          72.6 ±  7.6   34.5%  85.2%    0.0%     78
Liverpool         67.3 ±  7.6   13.2%  64.2%    0.0%     60
...
Wolves            36.9 ±  7.1    0.0%   0.0%   51.6%     20
Burnley           30.6 ±  6.7    0.0%   0.0%   83.1%     22
```

The `actual pts` column is the real season: the model called Arsenal's title
(38.6%, its favorite) but reality also produced genuine tail events — Man
United finished on 71 points (simulated xPts 52.2, top-4 probability 5.8%)
and Sunderland survived a 39.2% relegation probability by 15 points. A
±7.5-point standard deviation on every team's total is the honest headline:
single-season football is *noisy*, and a point forecast without a
distribution around it is close to meaningless.

### What-if analysis

Force any fixture to a chosen result and measure how every team's fate
shifts against the baseline:

```
$ ./build/simulate data/fixtures_2025_26.csv --sims 2000000 --force 327=A

forcing match 327: 2026-04-19  Man City vs Arsenal -> A (model had H 59% / D 21% / A 20%)

team               xPts     Δ |  title    Δpp |   top4    Δpp
Arsenal            75.5   +2.2 |  51.7%   +13.2 |  92.4%    +5.0
Man City           70.6   -2.0 |  22.8%   -11.7 |  79.1%    -6.1
Liverpool          67.3   +0.0 |  12.6%    -0.7 |  64.4%    +0.1
```

One April result at the Etihad swings the title race by ~25 percentage
points combined, while everyone outside the top two barely moves.
Implementation note: the hot loop has no concept of "forced" — conditioning
is done by editing that fixture's distribution to put all probability mass
on one outcome (`force_outcome` in `types.hpp`).

## Performance

Measured on an Apple M2 (4 performance + 4 efficiency cores), Apple clang 14,
`-O3`. Methodology: untimed warm-up run, then median of 3 (`bench/benchmark.cpp`).

**Generator shootout** — identical hot loop, generator swapped via template
parameter:

| generator    | seasons/sec | relative |
|--------------|-------------|----------|
| mt19937_64   |     131,221 | 1.00x    |
| xoshiro256++ |     241,400 | **1.84x** |

With ~400 random draws per season, generator cost is first-order.
xoshiro256++ (32 bytes of state, a few xor/shift/rotate ops per draw) beats
the Mersenne Twister (2.5 KB of state) by 1.84x at equal statistical quality
for this purpose — so the engine uses it. The decision is reproducible:
run `./build/benchmark`.

**Thread scaling** — full engine, 1M seasons:

| threads | seasons/sec | speedup | efficiency |
|---------|-------------|---------|------------|
|       1 |     239,307 |   1.00x |       100% |
|       2 |     462,293 |   1.93x |        97% |
|       4 |     848,522 |   3.55x |        89% |
|       8 |   1,092,879 |   4.57x |        57% |

**1.09M seasons/sec** end to end — 10 million full seasons in 9.3 s. Scaling
is near-perfect to 4 threads (the M2's performance cores), then flattens as
work spills onto efficiency cores: 8x on a 4P+4E chip is physically
impossible, and reporting 4.57x with the explanation beats pretending
otherwise. Combined with the RNG swap, total throughput improved **8.6x**
over the first correct implementation — 1.84x from measuring, 4.57x from
threading.

## Design

The library (`include/simulator/`, `src/`) is built around a few deliberate
decisions:

- **Hot/cold data separation.** A fixture is 12 bytes: two `uint16_t` team
  ids and a two-value CDF (`float`). Team names, dates, actual results live
  in separate arrays the simulation never touches; the whole season's
  working set is ~4.5 KB — a fraction of L1 cache. No strings, pointers, or
  allocations exist anywhere in the hot loop.
- **CDF precomputation.** Probabilities are stored pre-accumulated, so
  sampling a match outcome is one uniform draw and two comparisons.
- **Mergeable accumulator.** Statistics are integer counts (a rank
  histogram per team, plus points sums) — exact, no float drift, and two
  accumulators combine by addition. That property *is* the threading
  design: each thread fills its own, merged after `join()`. No mutex or
  atomic appears in the codebase.
- **Per-thread RNG streams.** Each worker owns a generator seeded with
  (user seed, thread index). Same seed + same thread count → bit-identical
  results, independent of OS scheduling, because work is split statically.
- **Templated hot loop.** `simulate_season<URBG>` is generic over the
  generator (any *UniformRandomBitGenerator*), which is what made the
  benchmark shootout a one-variable experiment and costs nothing at
  runtime — each instantiation inlines the generator into the loop.
- **Random tie-breaking, on purpose.** The model predicts outcomes, not
  scorelines, so goal difference is unmodeled. Points ties are broken
  uniformly at random (shuffle + stable sort — the shuffle survives within
  tied groups), which is symmetric across teams rather than inventing fake
  goals.

## Correctness

Monte Carlo bugs don't crash — they quietly skew distributions. The test
suite (`tests/test_main.cpp`, plain asserts, no framework) checks against
things known *exactly*:

- degenerate leagues with deterministic outcomes (exact points, zero variance)
- an all-draws league where ranks come only from the tie-break, which must
  be uniform
- analytically computed expected points, E[pts] = Σ 3·P(win) + P(draw)
  (law of large numbers, with asymmetric probabilities so index bugs can't
  cancel)
- structural invariants: one team per rank, one rank per team, every season
- determinism (same seed → bit-identical) and RNG-stream independence
- lossless accumulator merge, and single- vs multi-threaded statistical
  agreement
- forced outcomes occurring in 100% of simulated seasons

## Limitations

- Matches are simulated **independently** with pre-season probabilities: no
  form update after a simulated result, no injuries, no title-race
  psychology. The simulation propagates the model's uncertainty; it doesn't
  add football knowledge the model lacks.
- Outcome-level only — no scorelines, so no goal-difference tiebreaker
  (see above).
- Probabilities are frozen at export time; re-exporting mid-season with
  updated form is supported by the pipeline but not automated.

## Possible extensions

- **Dynamic ratings:** update team strength after each simulated match
  (Elo-style) so a simulated slump compounds — turns independence into a
  Markov chain.
- **Poisson scoreline model:** have Python export expected goals instead of
  outcome probabilities; simulate scores, enabling real goal-difference
  tiebreaks and "record points total" questions.
- **SIMD batch sampling:** draw and classify multiple fixtures per
  instruction with NEON/AVX intrinsics.
