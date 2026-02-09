---
name: benchmark
description: Guide users through rigorous ML/AI benchmarking experiments using xetrack. Use when users want to: (1) Compare ML models, hyperparameters, or architectures, (2) Benchmark LLM prompts, few-shot examples, or generation strategies, (3) Evaluate data processing pipelines or embeddings, (4) Set up reproducible experiments with caching and validation, (5) Debug existing benchmarks for data leaks or inconsistencies, (6) Analyze benchmark results with SQL/DuckDB. Helps design experiments end-to-start following single-execution principles.
---

# Benchmark Skill

Guide users through methodologically rigorous ML/AI benchmarking using xetrack, from experiment design to analysis.

## Core Philosophy

**Design experiments end-to-start.** Build robust single-execution functions first, cache aggressively, save raw responses, and version everything. The goal: implement once, run once, analyze without pain.

> *"If you get the unglamorous parts right—experiment design, execution, logging, and failure handling—analysis becomes almost boring. If you don't, every surprising result triggers a debugging session instead of insight."*

**Critical principle:** Design experiments so that **contradictory outcomes are possible and informative**, not just confirmatory. Avoid confirmation bias—your setup should be robust enough to distinguish genuine measurements from genuine discoveries.

## Prerequisites

**CRITICAL:** Before starting any benchmark, ensure xetrack is properly set up and understood:

### Step 0: Setup & Learn xetrack

1. **Verify git branch:**
   ```bash
   git branch --show-current
   # Must be on a feature branch (e.g., feat/experiment-name), not main!
   # If on main: git checkout -b feat/benchmark-experiment
   ```

2. **Install xetrack and dependencies:**
   ```bash
   pip install xetrack[duckdb,cache,assets]
   pip install polars  # For large dataset analysis (optional)
   ```

3. **Read xetrack documentation:**
   - **MUST READ**: Find the installed xetrack package and read its README for full API docs. Locate it with: `python -c "import xetrack; print(xetrack.__file__)"`
   - **Run example**: Verify installation works: `python -c "from xetrack import Tracker; t = Tracker('/tmp/test.db'); t.log({'x': 1}); print('OK')"`

4. **Understand core concepts:**
   - `Tracker(db, params=dict, engine='sqlite|duckdb', table='predictions')` - Main tracking interface
   - `tracker.log(dict)` - Log arbitrary data
   - `tracker.track(func, args, kwargs)` - Track function execution with auto-unpacking
   - `Reader(db, engine='sqlite|duckdb', table='predictions')` - Read tracked data
   - `xt` CLI - Command-line interface for queries

**⚠️ IMPORTANT:** xetrack is not a common package. DO NOT hallucinate APIs. Always reference README/examples for correct usage patterns.

## When to Use the Git Versioning Skill Instead

This benchmark skill handles **experiment design, execution, and analysis**. For **experiment versioning, parallel workflows, and result merging**, consider the `git-versioning` skill.

**Use this benchmark skill when:**
- Designing and running benchmarks (sequential or parallel parameter sweeps)
- Experiments only differ in parameters (same code/data) → use DuckDB engine with threads for parallel writes
- You need caching, validation, and analysis guidance

**Use the `git-versioning` skill when:**
- Parallel experiments need **different code or data** → git worktree workflow
- You need to **merge or rebase** results from multiple experiment branches
- You need **DVC pipeline integration** for reproducible merge/rebase operations
- You need to manage **model candidates and promotion** across experiments
- You want full **Git + DVC + xetrack** versioning with tags and reproducibility

**Quick decision:** Do your parallel experiments need code changes? **Yes → `git-versioning` skill.** No → stay here, use DuckDB engine for parallel.

---

## Workflow

Follow this sequential workflow. Read the referenced docs for detailed guidance on each phase.

### Development vs Experiment Phase

**IMPORTANT:** Benchmarking has two distinct phases. Do NOT mix them!

- **Development phase**: Build and test your pipeline (small subsets, rapid iteration, deletable runs)
- **Experiment phase**: Run real benchmarks (no code changes, full data, reproducible, git tagged)
- **Code change during experiment** = go back to development phase

For detailed testing requirements, error handling patterns, data inspection during development, and the full dev/experiment phase checklist:

```
Read references/dev-phase.md
```

### Phase 0: Ideation — Plan What to Track

**Do this BEFORE writing any code.** Answer three questions:
1. What questions do you want to answer?
2. What data do you need per datapoint?
3. What segmentations and comparisons will you perform?

### Phase 1: Design — Understand Goals & Design Experiment

**Start from the end.** Define what success looks like, what tables you need, and what parameters to track.

```
Read references/design.md    # Full Phase 0 + Phase 1 guidance
```

### Phase 2: Build — Single-Execution Function

**Critical principle: Every datapoint executes exactly once.** Build a frozen dataclass for params, a stateless prediction function, and two-table pattern (predictions + metrics).

### Phase 3: Cache — Add Caching

**Caching is a correctness tool**, not optimization. Prevents duplicate executions and wasted compute.

### Common Pitfalls

Key pitfalls to watch for: dataclass unpacking only works with `.track()`, mutable defaults in dataclasses, wrong cache directory, missing error handling, and schema drift.

```
Read references/build-and-cache.md    # Full Phase 2 + Phase 3 + Pitfalls
```

### Phase 4: Parallelize (Optional)

If benchmark is slow, parallelize after validating single-execution. Each worker creates its own tracker. Use DuckDB with `ThreadPoolExecutor` for I/O-bound work, or SQLite with `multiprocessing.Pool` for CPU-bound work.

### Phase 5: Run Full Benchmark Loop

Run the complete experiment with progress tracking, metrics aggregation, and error handling.

```
Read references/execution.md    # Full Phase 4 + Phase 5 guidance
```

### Phase 6: Validate Results

Before analysis, check for data leaks, missing parameters, and failed executions.

### Phase 7: Analyze with DuckDB

xetrack + DuckDB = powerful analysis. Compare experiments, generate summaries, export results.

### Schema Validation

**Critical check:** Detect parameter renames or schema drift before running experiments. If you rename a parameter in code, xetrack creates a NEW column instead of reusing the old one.

```
Read references/analysis.md    # Full Phase 6 + Phase 7 + Schema Validation code
```

---

## Common Patterns by Use Case

### Pattern 1: sklearn Model Comparison

See `assets/sklearn_benchmark_template.py` for complete example.

**Key points:**
- Use frozen dataclass for hyperparameters
- Cache fitted models in xetrack assets (requires `pip install xetrack[assets]`)
- Track both training and test metrics

### Pattern 2: LLM Prompt Benchmarking

See `assets/llm_finetuning_template.py` for complete example.

**Key points:**
- Save full LLM response (not just parsed output)
- Track token counts, cost, latency
- Cache responses to avoid re-querying
- Handle failures gracefully (rate limits, timeouts)

### Pattern 3: Load Testing / Throughput

See `assets/throughput_benchmark_template.py` for complete example.

**Key points:**
- Use `log_system_params=True` for CPU/memory tracking
- Measure requests per second, p50/p95/p99 latency
- Simulate concurrent load with multiprocessing

---

## Helper Scripts

All scripts are in `scripts/`:

- **`validate_benchmark.py`** - Check for data leaks, duplicate executions
- **`analyze_cache_hits.py`** - Analyze caching effectiveness
- **`export_summary.py`** - Generate markdown summaries

Usage:
```bash
python scripts/validate_benchmark.py <db_path> <table_name>
```

---

## Data & Database Versioning

For full experiment versioning with DVC, git tags, worktree workflows, merge/rebase semantics, and artifact retrieval, use the **`git-versioning` skill**. It covers DVC setup, sequential/parallel/worktree workflows, model management, and experiment exploration.

### Minimal Versioning (within this skill)

For quick benchmarks that don't need full DVC, track git state in your params:

```python
import subprocess

tracker = Tracker(
    db='benchmark.db',
    engine='duckdb',
    params={
        'git_hash': subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip(),
        'experiment_version': 'e0.0.1',
    }
)
```

### Database Management Patterns

**Pattern 1: Append-Only** (simplest, recommended to start)
- Single `benchmark.db`, new experiments append rows
- Filter by `experiment_version` column in queries

**Pattern 2: Table-per-Experiment** (for >10 experiments or >100MB)
```python
tracker = Tracker(db='benchmark.db', table=f'predictions_{experiment_name}')
```

**Pattern 3: Database-per-Experiment** (clean separation)
```python
tracker = Tracker(db=f'{experiment_name}.db', table='predictions')
```

---

## Git Tag-Based Experiment Versioning

For comprehensive git tag workflows, DVC integration, and worktree-based parallel experiments, see the **`git-versioning` skill**. Below is the minimal pattern for tagging within benchmarks:

### Quick Tagging Pattern

Track `experiment_version` in your params, then tag after each experiment:

```python
tracker = Tracker(
    db='benchmark.db',
    engine='duckdb',
    table='predictions',
    params={'experiment_version': 'e0.0.1'}
)

# After experiment completes:
# git add benchmark.db && git commit -m "experiment: e0.0.1"
# git tag -a e0.0.1 -m "model=bert | acc=0.85 | baseline"
```

For the full tag workflow (auto-increment, DVC integration, tag description generation, commit hash tracking), see the `git-versioning` skill's `scripts/version_tag.py`.

For pre-run data validation, DVC commit checks, hash tracking, and experiment history queries, see the **`git-versioning` skill** which provides `scripts/version_tag.py` and `scripts/experiment_explorer.py` for these workflows.

---

## References

For deeper guidance, see:

**Phase-specific references:**
- **`references/dev-phase.md`** - Development vs experiment phase: testing, error handling, data inspection
- **`references/design.md`** - Phase 0 (Ideation) + Phase 1 (Design): planning what to track, experiment design
- **`references/build-and-cache.md`** - Phase 2 (Build) + Phase 3 (Caching) + Common Pitfalls
- **`references/execution.md`** - Phase 4 (Parallelize) + Phase 5 (Run Full Benchmark Loop)
- **`references/analysis.md`** - Phase 6 (Validate) + Phase 7 (DuckDB Analysis) + Schema Validation

**General references:**
- **`references/methodology.md`** - Core benchmarking principles and philosophy
- **`references/duckdb-analysis.md`** - DuckDB queries and analysis recipes

---

## Quick Start: Complete Minimal Example

**30-line end-to-end benchmark** showing two-table pattern:

```python
from dataclasses import dataclass
from xetrack import Tracker, Reader

# 1. Define params as frozen dataclass
@dataclass(frozen=True, slots=True)
class ModelParams:
    model: str
    threshold: float = 0.5

# 2. Single-execution function
def predict(item, params):
    prediction = item['x'] > params.threshold
    return {
        'input_id': item['id'],
        'prediction': prediction,
        'ground_truth': item['label'],
        'confidence': abs(item['x'] - params.threshold)
    }

# 3. Create trackers (two tables: predictions + metrics)
pred_tracker = Tracker(db='bench.db', engine='sqlite', cache='cache', table='predictions')
metrics_tracker = Tracker(db='bench.db', engine='sqlite', table='metrics')

# 4. Run benchmark
dataset = [{'id': 1, 'x': 0.3, 'label': False}, {'id': 2, 'x': 0.7, 'label': True}]
params = ModelParams(model='baseline', threshold=0.5)

results = [pred_tracker.track(predict, args=[item, params]) for item in dataset]

# 5. Calculate and log metrics
accuracy = sum(r['prediction'] == r['ground_truth'] for r in results) / len(results)
metrics_tracker.log({'model': params.model, 'threshold': params.threshold, 'accuracy': accuracy})

# 6. Analyze
print("Predictions:", Reader(db='bench.db', engine='sqlite', table='predictions').to_df())
print("Metrics:", Reader(db='bench.db', engine='sqlite', table='metrics').to_df())
```

**Why this example uses SQLite:** Safe for any scenario (single-process or multiprocessing).

Done! Now run the validation scripts and start analyzing.

---

## When NOT to Use This Workflow

Not every benchmark needs this level of rigor. **Skip this workflow** when:

**✅ Use simpler approach:**
- **Quick one-off comparison** (< 5 minutes to re-run everything)
- **Early prototyping phase** (speed of iteration > reproducibility)
- **Small-scale experiments** (< 100 datapoints, re-running is cheap)
- **Solo exploration** (no team coordination, throwaway analysis)

**🚫 Use this workflow:**
- **Results will be shared or published**
- **Experiment takes > 10 minutes to run**
- **Testing expensive APIs** (LLMs, cloud services, paid APIs)
- **Team collaboration** (multiple people running experiments)
- **Production benchmarks** (decisions depend on results)
- **Reproducibility matters** (research, A/B tests, audits)

**Key principle:** Match infrastructure complexity to problem complexity. Over-engineering wastes more time than it saves.

**Remember:**
> *"Storage is cheap; regret is expensive. The data you didn't save is the question you'll care about most."*

When in doubt, err on the side of tracking more.

---

## Alternative Tools & Platforms

While this skill focuses on xetrack + DuckDB/SQLite, consider these alternatives for different needs:

**Experiment Tracking Platforms:**
- **[Weights & Biases](https://wandb.ai/)**: Cloud-based, great for teams, rich visualizations
- **[MLflow](https://mlflow.org/)**: Open-source, self-hosted, model registry
- **[Neptune](https://neptune.ai/)**: Metadata store, experiment comparison
- **[Aim](https://github.com/aimhubio/aim)**: Lightweight, self-hosted, focuses on metrics

**Caching Libraries:**
- **[joblib](https://joblib.readthedocs.io/)**: Function result caching for complex objects
- **[diskcache](https://grantjenks.com/docs/diskcache/)**: Disk-based cache (what xetrack uses)
- **[GPTCache](https://github.com/zilliztech/GPTCache)**: LLM-specific semantic caching

**When to use alternatives:**
- Large teams → W&B or MLflow (better collaboration features)
- Complex pipelines → MLflow (pipeline tracking, model registry)
- LLM-heavy → GPTCache (semantic similarity caching)
- Need hosted solution → W&B or Neptune (no infrastructure management)

**When xetrack + DuckDB/SQLite wins:**
- Solo or small team
- Want full control and ownership of data
- Need SQL flexibility for custom analysis
- Prefer local-first, no cloud dependencies
- Git-based versioning workflow
