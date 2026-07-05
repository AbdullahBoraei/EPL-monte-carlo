# epl-monte-carlo

A multithreaded Monte Carlo season simulator in C++20 with zero dependencies.
It takes per-match outcome probabilities from my
[EPL match predictor](https://github.com/AbdullahBoraei/EPL-match-prediction)
(Python, logistic regression) and plays out the full Premier League season
more than a million times per second. It answers season-level questions:

- each team's probability of winning the title, reaching the top 4, or going down
- the expected final table, with a spread around each team's points total
- the shift in every team's chances after you pin one match to a chosen result

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

Each language does the work it suits: Python owns the modeling (pandas,
scikit-learn) and C++ owns the compute-bound simulation. A plain CSV crosses
the boundary in each direction, so you can swap either side without touching
the other. `scripts/export_fixtures.py` in the predictor repo produces the
input, and `data/` ships a copy for 2025-26, so this repo runs on its own.

## Quick start

```bash
cmake -B build -S .        # Release (-O3) by default
cmake --build build
./build/tests              # correctness suite
./build/simulate data/fixtures_2025_26.csv --sims 1000000 --out results.csv
```

Sample output (1M simulated 2025-26 seasons, from probabilities the model
set before the season):

```
team                    xPts    title    top4   releg actual pts
Arsenal           73.3 ±  7.5   38.6%  87.4%    0.0%     85
Man City          72.6 ±  7.6   34.5%  85.2%    0.0%     78
Liverpool         67.3 ±  7.6   13.2%  64.2%    0.0%     60
...
Wolves            36.9 ±  7.1    0.0%   0.0%   51.6%     20
Burnley           30.6 ±  6.7    0.0%   0.0%   83.1%     22
```

The `actual pts` column shows the real season. The model made Arsenal its
title favorite at 38.6%, and Arsenal won the league. The season also produced
tail outcomes: Man United finished on 71 points against a simulated 52.2
expected points and a 5.8% top-4 probability, and Sunderland outlived a 39.2%
relegation probability by 15 points. Each team's simulated total carries a
standard deviation near 7.5 points; a single season leaves that much to
chance, and a point forecast hides that spread.

### What-if analysis

Pin any fixture to a chosen result and measure the shift in every team's
chances against the baseline:

```
$ ./build/simulate data/fixtures_2025_26.csv --sims 2000000 --force 327=A

forcing match 327: 2026-04-19  Man City vs Arsenal -> A (model had H 59% / D 21% / A 20%)

team               xPts     Δ |  title    Δpp |   top4    Δpp
Arsenal            75.5   +2.2 |  51.7%   +13.2 |  92.4%    +5.0
Man City           70.6   -2.0 |  22.8%   -11.7 |  79.1%    -6.1
Liverpool          67.3   +0.0 |  12.6%    -0.7 |  64.4%    +0.1
```

One April result at the Etihad moves the top two's title probabilities by a
combined 25 percentage points, while every other team shifts by less than
one. The hot loop contains zero code for forced matches: `force_outcome`
in `types.hpp` edits that fixture's distribution so the chosen outcome
carries the full probability mass.

## Performance

I measured on an Apple M2 (4 performance + 4 efficiency cores) with Apple
clang 14 at `-O3`. Each configuration runs after an untimed warm-up, and the
tables report the median of three runs (`bench/benchmark.cpp`).

**Generator comparison.** The templated hot loop lets the benchmark swap the
generator and hold every other instruction fixed:

| generator    | seasons/sec | relative |
|--------------|-------------|----------|
| mt19937_64   |     131,221 | 1.00x    |
| xoshiro256++ |     241,400 | 1.84x    |

A season consumes 380 match draws plus a 20-element shuffle, so generator
cost sets the pace. xoshiro256++ keeps 32 bytes of state and updates it with
a handful of xor, shift, and rotate operations, while std::mt19937_64
carries 2.5 KB. Both pass the standard statistical test suites for this
workload, so I picked xoshiro256++ as the engine default given the 1.84x
margin. Run `./build/benchmark` to reproduce the decision.

**Thread scaling.** Full engine, 1M seasons:

| threads | seasons/sec | speedup | efficiency |
|---------|-------------|---------|------------|
|       1 |     239,307 |   1.00x |       100% |
|       2 |     462,293 |   1.93x |        97% |
|       4 |     848,522 |   3.55x |        89% |
|       8 |   1,092,879 |   4.57x |        57% |

The engine reaches 1.09M seasons per second, or 10 million full seasons in
9.3 s. Efficiency holds at 89% and above through 4 threads, which matches
the M2's count of performance cores; past that, work spills onto the
efficiency cores and the curve flattens. The 4P+4E topology caps the
achievable speedup below 8x, and the table reports the measured 4.57x
together with that cause. The generator swap and the threading combine for
an 8.6x gain over the first correct implementation.

## Design

- **Hot/cold data separation.** A fixture occupies 12 bytes: two `uint16_t`
  team ids and a two-value CDF in `float`. Separate arrays hold team names,
  dates, and actual results, and the simulation reads none of them. The
  season's working set spans 4.5 KB and fits in a slice of L1 cache,
  and the hot loop touches flat arrays of integers and floats from start to
  finish.
- **CDF precomputation.** The loader stores each fixture's probabilities
  pre-accumulated, so one uniform draw and two comparisons decide a match.
- **Mergeable accumulator.** The statistics live as integer counts (a rank
  histogram per team plus points sums), which stay exact over millions of
  seasons, and two accumulators combine by addition. That property carries
  the threading design: each thread fills its own accumulator, the engine
  merges them after `join()`, and the hot path synchronizes at that single
  point.
- **Per-thread RNG streams.** Each worker owns a generator seeded with the
  user seed and its thread index. A run with a given seed and thread count
  reproduces bit for bit, independent of OS scheduling, because the engine
  splits the work into fixed shares.
- **Templated hot loop.** `simulate_season<URBG>` accepts any
  *UniformRandomBitGenerator*, which made the generator comparison a
  one-variable experiment at zero runtime cost: each instantiation inlines
  the generator into the loop.
- **Random tie-breaking by design.** The model predicts three-way outcomes,
  so goal difference sits outside the simulation. Points ties
  resolve by a draw with equal probability across the tied teams (a shuffle
  followed by a stable sort, which preserves the shuffled order inside tied
  groups), and the symmetry protects every team from an artifact advantage.

## Correctness

A bug in a Monte Carlo engine tends to skew distributions while the program
keeps running, so the test suite (`tests/test_main.cpp`, plain assertions)
checks results against values known in closed form:

- leagues with probabilities of 0 and 1, where points and ranks follow with
  zero variance
- an all-draws league, where the tie-break decides every rank and each team
  must finish first in an equal share of seasons
- expected points computed by hand from E[pts] = Σ 3·P(win) + P(draw), which
  the simulated means must approach; the test uses asymmetric probabilities
  so an index bug would shift the answer and fail the check
- structural invariants: each simulated season places one team at each rank
  and each team at one rank
- reproducibility: a repeated seed yields identical counts, and a changed
  seed yields different ones
- lossless accumulator merges, plus statistical agreement between
  single-threaded and multithreaded runs
- forced outcomes occurring in every simulated season

## Limitations

- The engine simulates matches as independent events with pre-season
  probabilities: a simulated result leaves the following matches'
  probabilities unchanged, so form swings, injuries, and title-race pressure
  stay outside the model. The simulation propagates the model's uncertainty
  and contributes zero football knowledge of its own.
- The engine works at the outcome level; scorelines and goal-difference
  tiebreaks require an expected-goals model (see the extensions below).
- Probabilities freeze at export time. The pipeline supports a mid-season
  re-export with updated form, and automating that step remains future work.

## Possible extensions

- **Dynamic ratings:** update team strength after each simulated match, Elo
  style, so a simulated slump compounds; this turns independent matches into
  a Markov chain.
- **Poisson scoreline model:** export expected goals from Python in place of
  outcome probabilities and simulate scores, which enables goal-difference
  tiebreaks and questions about record points totals.
- **SIMD batch sampling:** draw and classify several fixtures per
  instruction with NEON or AVX intrinsics.
