# Test budget guard

`hwoutils.testbudget` is a pytest plugin that keeps a JAX test suite cheap. It
fails or reports tests that run too long, counts JAX compiles, watches resident
memory with a hard ceiling, optionally clears JAX caches between modules, and
appends one summary line per session to a trend file.

## Adopting it

Add the plugin to the top-level `conftest.py`:

```python
pytest_plugins = ["hwoutils.testbudget.plugin"]
```

Add a budget table to `pyproject.toml`. Start in report mode so nothing fails
while the suite is measured:

```toml
[tool.testbudget]
mode = "report"          # "enforce" fails the offending test
unit_s = 5.0
medium_s = 30.0
slow_s = 300.0
rss_ceiling_gb = 12.0
clear_caches = true
```

Run the suite with `--budget-report` to see the slowest tests, tests that grew
resident memory, and compiles per module. `--testbudget-mode report` or
`--testbudget-mode enforce` overrides the file for one run, and
`--testbudget-off` disables the guard.

## Tiers

Each test carries one tier. A test with no marker is unit.

| Tier | Marker | Call-time budget | Meaning |
|---|---|---|---|
| unit | none | `unit_s` (5 s, includes first compile) | tiny grids, closed forms, contracts, regressions |
| medium | `@pytest.mark.medium` | `medium_s` (30 s) | integration across modules, small convergence ladders |
| slow | `@pytest.mark.slow` | `slow_s` (300 s) | production-size grids and campaigns |

A test over its tier budget fails in enforce mode with a message naming the
time and the budget. The remedy is to shrink the test or reuse a compiled
fixture, not to raise the budget. `@pytest.mark.time_budget(seconds)`
overrides the tier for one test and should carry a comment giving the reason.

## Memory

The plugin samples resident memory for each process (each xdist worker is its
own process). A watchdog thread polls every `rss_poll_s` seconds and exits the
process with code 97 when it passes `rss_ceiling_gb`, in any mode, so a runaway
test stops long before the machine freezes. `slow_rss_ceiling_gb` sets a higher
ceiling while a slow test runs. The report lists every test whose resident
growth reaches `rss_growth_report_mb`.

## Compiles

The plugin counts XLA programs compiled in each test and adds up the backend
compile seconds. Only real compiles count: a program loaded from the persistent
cache is excluded from both the count and the seconds. A warm cache therefore
reports little or no compile cost, so compile budgets are meaningful on a cold
cache: a cold CI runner, or a run with `TESTBUDGET_CACHE_DIR` pointed at an
empty directory. Trend counts recorded before this accounting are not
comparable, because a cache hit used to count as 2. The numbers come from the
JAX monitoring events `/jax/core/compile/backend_compile_duration` (which also
fires for a cache load) and `/jax/compilation_cache/cache_hits` (which marks
that load). JAX records no event that only a real compile emits, so the plugin
pairs each cache-hit event with the duration event that follows it on the same
thread, and clears the pytest thread's pending hits at the start of every test. A cache
load running on a background thread across a test boundary still pairs
correctly. The one assumption is that other monitoring listeners neither raise
nor compile inside a compile; a raising listener makes the compiling call
raise, so the test errors visibly, and it can leave an unpaired hit on a
background thread, which the plugin does not clean up.
The summary line ends with `compile_s=`, and `--budget-report` lists compile
count and seconds per module, top ten by seconds.

- `@pytest.mark.compile_budget(n)` fails a test that compiles more than `n`
  programs. Use it as a cache-hit test: call a jitted entry point with two
  value-different objects and require one compile.
- `module_compile_budget` and `module_compile_budget_by_file` cap the compiles
  in a test module. A module over budget reports an error in the teardown of
  its last test (enforce mode). A jump usually means a numeric value became a
  static jit key.
- Modules are keyed by their path relative to the pytest rootdir, for example
  `tests/test_foo.py`, so two files with the same name in different
  directories keep separate totals. In the `_by_file` mappings a key may be
  that relative path or the bare file name; the relative path wins when both
  match. A module that pytest enters more than once (reordered tests) is judged
  on its cumulative total, the same under xdist.
- `module_compile_seconds` and `module_compile_seconds_by_file` cap the compile
  seconds in a test module the same way. Count is a weak proxy, since a 0.1 s
  compile and an 18 s compile count the same, so a seconds budget catches a
  regression that adds a few expensive programs. The budget is off by default,
  follows `mode` (reported in `report`, an error in the last test's teardown in
  `enforce`), and is used as written: the plugin applies no scale factor to any
  budget, so set a looser value for a slower runner. Under xdist the module
  counts and seconds are summed over workers and checked once at the end of the
  session, so a module split across workers is judged on its total; in enforce
  mode the session then exits nonzero and lists the module under over-budget
  items, because no test is left to fail.

## Fixture and compilation cache

With `clear_caches = true` an autouse module-scoped fixture calls
`jax.clear_caches()` and `gc.collect()` after each module. Set
`compile_cache = true` to enable the persistent compilation cache in
`compile_cache_dir` (default `.jax_cache` under the project root; a
`.gitignore` containing `*` is written inside). `TESTBUDGET_CACHE_DIR`
overrides the directory.

## Trend file

Each session appends a JSON line (library, commit, invocation, tests, wall
time, peak resident GB, compiles, compile seconds, violations, slowest five) to
`~/hwo-data/testbudget/<library>.jsonl`. `trend_file` or
`TESTBUDGET_TREND_FILE` changes the path, and `trend = false` turns it off.

```bash
testbudget trend hwoutils            # a library name or a path to a .jsonl file
```

The command prints the latest runs and flags any metric (wall time, peak
resident memory, compiles, test count) more than 20 percent above the mean of
the previous 10 runs with the same invocation. It exits 1 when something is
flagged.

## Configuration keys

All keys live under `[tool.testbudget]`. An unknown key is an error.

| Key | Default | Meaning |
|---|---|---|
| `library` | project name | trend file name |
| `mode` | `report` | `report` or `enforce` |
| `unit_s`, `medium_s`, `slow_s` | 5, 30, 300 | call-time budgets in seconds |
| `rss_ceiling_gb` | 12 | per-process hard abort; 0 disables |
| `slow_rss_ceiling_gb` | 0 | ceiling during slow tests; 0 means the same as above |
| `rss_poll_s` | 0.5 | watchdog period |
| `rss_growth_report_mb` | 500 | growth threshold for the report |
| `module_compile_budget` | 0 | compiles per module; 0 disables |
| `module_compile_budget_by_file` | empty | per-file overrides |
| `module_compile_seconds` | 0 | compile seconds per module; 0 disables |
| `module_compile_seconds_by_file` | empty | per-file overrides |
| `clear_caches` | false | clear JAX caches after each module |
| `compile_cache`, `compile_cache_dir` | false, `.jax_cache` | persistent cache |
| `trend`, `trend_file` | true, empty | trend file switch and path |
