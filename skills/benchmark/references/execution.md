# Execution

Phase 4 (Parallelize) and Phase 5 (Run Full Benchmark Loop). Referenced from SKILL.md.

## Phase 4: Parallelize (Optional)

If benchmark is slow, parallelize after validating single-execution.

**⚠️ Engine choice matters for parallelism:**

| Method | DuckDB | SQLite |
|--------|--------|--------|
| `ThreadPoolExecutor` | ✅ Works | ✅ Works |
| `multiprocessing.Pool` | ❌ File locks | ✅ Works (WAL mode) |

### Option A: Threading with DuckDB (recommended for most cases)

```python
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Stateless function is safe to parallelize
def run_with_tracker(item, params):
    tracker = Tracker(db='benchmark.db', engine='duckdb',
                     cache='cache_dir', table='predictions')
    return tracker.track(run_single_prediction, args=[item, params])

# Parallel execution with threads
params = BenchmarkParams(model_name='bert-base', embedding_dim=768)
with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(partial(run_with_tracker, params=params), data))
```

### Option B: Multiprocessing with SQLite (for CPU-bound work)

```python
from multiprocessing import Pool
from functools import partial

# Stateless function is safe to parallelize
def run_with_tracker(item, params):
    tracker = Tracker(db='benchmark.db', engine='sqlite',
                     cache='cache_dir', table='predictions')
    return tracker.track(run_single_prediction, args=[item, params])

# Parallel execution with processes
params = BenchmarkParams(model_name='bert-base', embedding_dim=768)
with Pool(processes=4) as pool:
    results = pool.map(partial(run_with_tracker, params=params), data)
```

**Important:** DuckDB uses file-level locks that block concurrent access from separate processes. Use SQLite (WAL mode) for multiprocessing, or DuckDB with threads.

### When to Use Which

| Workload | Example | Recommendation |
|----------|---------|----------------|
| **I/O-bound** (waiting on APIs, network, disk) | LLM API calls, embedding services, web scraping | Threads + DuckDB — threads release the GIL during I/O waits |
| **CPU-bound** (heavy local computation) | sklearn cross-validation, feature extraction, image processing | Multiprocessing + SQLite — separate processes bypass the GIL |
| **Not sure** | | Threads + DuckDB — safer default, simpler to debug |

---

## Phase 5: Run Full Benchmark Loop

### Pre-Experiment Validation & Design Review

**Critical:** Before EVERY experiment, validate and review your design.

You might discover new things you want to collect or new aggregations for the metrics table.

```python
import subprocess
from xetrack import Reader

def pre_experiment_validation(
    db_path='benchmark.db',
    predictions_table='predictions',
    metrics_table='metrics',
    params_dataclass=None
):
    """
    Comprehensive pre-experiment validation and design review.
    Run this before EVERY experiment, not just the first one.
    """
    print("\n" + "="*60)
    print("PRE-EXPERIMENT VALIDATION & DESIGN REVIEW")
    print("="*60 + "\n")

    issues = []

    # ========================================
    # 1. SCHEMA VALIDATION
    # ========================================
    print("1️⃣  SCHEMA VALIDATION")
    print("-" * 40)

    if not validate_schema_before_experiment(db_path, predictions_table, params_dataclass):
        issues.append("Schema validation failed")
    else:
        print("✅ Schema matches code\n")

    # ========================================
    # 2. CODE CHANGES SINCE LAST EXPERIMENT
    # ========================================
    print("2️⃣  CODE CONSISTENCY CHECK")
    print("-" * 40)

    try:
        # Get last experiment's code commit using xetrack Reader (engine-agnostic)
        reader = Reader(db_path, table=metrics_table)
        df = reader.to_df()
        result = df.sort_values('timestamp', ascending=False).iloc[0]['code_commit'] if len(df) > 0 else None

        if result is not None:
            last_code_commit = result
            current_commit = subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD']
            ).decode().strip()

            if last_code_commit != current_commit:
                print(f"⚠️  Code changed since last experiment!")
                print(f"   Last experiment: {last_code_commit}")
                print(f"   Current code:    {current_commit}\n")

                # Show what changed
                try:
                    diff = subprocess.check_output(
                        ['git', 'diff', '--stat', last_code_commit, 'HEAD'],
                        stderr=subprocess.DEVNULL
                    ).decode()
                    print("   Files changed:")
                    print("   " + "\n   ".join(diff.split('\n')[:10]))  # First 10 lines
                except:
                    pass

                print("\n   ⚠️  Code changes may affect results!")
                print("   Questions to consider:")
                print("   - Did you test the changes? (dev phase)")
                print("   - Are results still comparable to previous experiments?")
                print("   - Should this be a new experiment series?\n")

                response = input("   Continue with code changes? [y/N]: ").strip().lower()
                if response != 'y':
                    issues.append("User aborted due to code changes")
            else:
                print(f"✅ Code unchanged since last experiment ({current_commit})\n")
        else:
            print("ℹ️  First experiment - no previous code to compare\n")

    except Exception as e:
        print(f"⚠️  Could not check code changes: {e}\n")

    # ========================================
    # 3. DESIGN REVIEW - WHAT ARE WE TRACKING?
    # ========================================
    print("3️⃣  DESIGN REVIEW")
    print("-" * 40)

    # Show current schema using xetrack Reader (works with both SQLite and DuckDB)
    # Predictions table
    try:
        pred_reader = Reader(db_path, table=predictions_table)
        pred_df = pred_reader.to_df().head(0)  # Just get columns
        pred_columns = list(pred_df.columns)
        print(f"📊 Current predictions table columns ({len(pred_columns)}):")
        for col in sorted(pred_columns):
            print(f"   - {col}")
        print()
    except:
        print(f"ℹ️  Table '{predictions_table}' does not exist yet (first run)\n")
        pred_columns = []

    # Metrics table
    try:
        metrics_reader = Reader(db_path, table=metrics_table)
        metrics_df = metrics_reader.to_df().head(0)  # Just get columns
        metrics_columns = list(metrics_df.columns)
        print(f"📈 Current metrics table columns ({len(metrics_columns)}):")
        for col in sorted(metrics_columns):
            print(f"   - {col}")
        print()
    except:
        print(f"ℹ️  Table '{metrics_table}' does not exist yet (first run)\n")
        metrics_columns = []

    # ========================================
    # 4. DESIGN REFLECTION QUESTIONS
    # ========================================
    print("4️⃣  DESIGN REFLECTION")
    print("-" * 40)
    print("Based on previous experiments, do you want to add anything?\n")

    print("Questions to consider:")
    print("  • Are there NEW fields you wish you had tracked?")
    print("    (e.g., input_length, domain, difficulty, cost, confidence)")
    print()
    print("  • Are there NEW aggregations for metrics table?")
    print("    (e.g., p95_latency, error_rate, cost_per_correct_prediction)")
    print()
    print("  • Did you save RAW RESPONSES? Can you reprocess if needed?")
    print()
    print("  • Are there NEW segmentations you want to analyze?")
    print("    (e.g., performance by domain, by input length, by difficulty)")
    print()

    response = input("Do you want to ADD NEW FIELDS before this experiment? [y/N]: ").strip().lower()

    if response == 'y':
        print("\n📝 DESIGN UPDATE")
        print("-" * 40)
        print("Recommended workflow:")
        print("  1. Update your dataclass/params to include new fields")
        print("  2. Update prediction function to return new fields")
        print("  3. Test on small subset in DEV mode")
        print("  4. Return here and run validation again")
        print()
        print("Example:")
        print("  @dataclass(frozen=True)")
        print("  class ModelParams:")
        print("      model: str")
        print("      temperature: float")
        print("      domain: str  # ← NEW FIELD")
        print()
        print("  def predict(item, params):")
        print("      ...")
        print("      return {")
        print("          'raw_response': response,")
        print("          'prediction': parsed,")
        print("          'confidence': score,  # ← NEW FIELD")
        print("          'input_length': len(item['text'])  # ← NEW FIELD")
        print("      }")
        print()

        proceed = input("Continue to DEV mode to add fields? [y/N]: ").strip().lower()
        if proceed == 'y':
            issues.append("User wants to add fields - returning to DEV mode")

    # ========================================
    # 5. METRICS TABLE REVIEW
    # ========================================
    if metrics_columns:
        print("\n5️⃣  METRICS TABLE REVIEW")
        print("-" * 40)
        print("Current aggregations in metrics table:")
        for col in sorted(metrics_columns):
            if col not in ['timestamp', 'track_id', 'experiment_version']:
                print(f"   - {col}")
        print()

        response = input("Do you want to ADD NEW AGGREGATIONS to metrics? [y/N]: ").strip().lower()
        if response == 'y':
            print("\n📝 METRICS UPDATE")
            print("-" * 40)
            print("Common aggregations to consider:")
            print("  • Error rate: (errors / total) * 100")
            print("  • P95 latency: np.percentile(latencies, 95)")
            print("  • Cost per correct: total_cost / correct_predictions")
            print("  • Average confidence: mean(confidence_scores)")
            print("  • Failure modes: count by error_type")
            print()
            print("Update your metrics logging code:")
            print("  metrics_tracker.log({")
            print("      'accuracy': accuracy,")
            print("      'avg_latency': np.mean(latencies),")
            print("      'p95_latency': np.percentile(latencies, 95),  # ← NEW")
            print("      'error_rate': error_rate,  # ← NEW")
            print("      'cost_per_correct': total_cost / correct  # ← NEW")
            print("  })")
            print()

    # ========================================
    # 6. DATA COMMIT CHECK
    # ========================================
    print("\n6️⃣  DATA VERSION CHECK (Optional)")
    print("-" * 40)
    # check_data_committed()  # Uncomment if using DVC

    # ========================================
    # 7. FINAL DECISION
    # ========================================
    print("\n" + "="*60)
    if issues:
        print("❌ VALIDATION FAILED")
        print("="*60)
        for issue in issues:
            print(f"  • {issue}")
        print()
        return False
    else:
        print("✅ VALIDATION PASSED - READY FOR EXPERIMENT")
        print("="*60)
        print()
        return True

# Usage before every experiment:
if not pre_experiment_validation(
    db_path='benchmark.db',
    predictions_table='predictions',
    metrics_table='metrics',
    params_dataclass=ModelParams
):
    print("Experiment aborted. Fix issues or update design first.")
    exit(1)

print("🚀 Starting experiment...")
```

**Why validate before EVERY experiment:**

1. **You learn as you go** - First experiments reveal what you wish you had tracked
2. **Catch schema drift** - Parameter renames or new fields
3. **Code changes** - See what changed since last experiment
4. **Design evolution** - Add new aggregations to metrics table
5. **Maintain consistency** - Ensure experiments are comparable

**Example session:**

```
PRE-EXPERIMENT VALIDATION & DESIGN REVIEW
============================================================

1️⃣  SCHEMA VALIDATION
----------------------------------------
✅ Schema matches code

2️⃣  CODE CONSISTENCY CHECK
----------------------------------------
⚠️  Code changed since last experiment!
   Last experiment: a1b2c3d
   Current code:    e4f5g6h

   Files changed:
   model.py | 23 ++++++++++++++-------
   utils.py | 5 +++++

   ⚠️  Code changes may affect results!
   Questions to consider:
   - Did you test the changes? (dev phase)
   - Are results still comparable to previous experiments?
   - Should this be a new experiment series?

   Continue with code changes? [y/N]: y

3️⃣  DESIGN REVIEW
----------------------------------------
📊 Current predictions table columns (12):
   - cache
   - confidence
   - error
   - ground_truth
   - input_id
   - params_model
   - params_temperature
   - prediction
   - raw_response
   - timestamp
   - track_id

📈 Current metrics table columns (8):
   - accuracy
   - avg_latency
   - code_commit
   - data_commit
   - experiment_version
   - model
   - timestamp
   - track_id

4️⃣  DESIGN REFLECTION
----------------------------------------
Based on previous experiments, do you want to add anything?

Questions to consider:
  • Are there NEW fields you wish you had tracked?
    (e.g., input_length, domain, difficulty, cost, confidence)

  • Are there NEW aggregations for metrics table?
    (e.g., p95_latency, error_rate, cost_per_correct_prediction)

  • Did you save RAW RESPONSES? Can you reprocess if needed?

  • Are there NEW segmentations you want to analyze?
    (e.g., performance by domain, by input length, by difficulty)

Do you want to ADD NEW FIELDS before this experiment? [y/N]: y

📝 DESIGN UPDATE
----------------------------------------
Recommended workflow:
  1. Update your dataclass/params to include new fields
  2. Update prediction function to return new fields
  3. Test on small subset in DEV mode
  4. Return here and run validation again
...
```

This interactive validation ensures you don't run expensive experiments only to realize later you forgot to track something important.

### Recommended Workflow: Debug Small, Then Scale

**Best practice:** Start with a small subset to debug your pipeline before running the full benchmark.

```python
# 1. DEBUG MODE: Test with small subset first
DEBUG = True  # Set to False for full run
dataset_subset = dataset[:10] if DEBUG else dataset  # Only 10 items for testing

# Run benchmark on subset
for item in dataset_subset:
    result = predictions_tracker.track(run_prediction, args=[item, params])

# 2. Check results look correct
print(f"Processed {len(dataset_subset)} items")
reader = Reader(db='benchmark.db', table='predictions')
print(reader.to_df().tail())

# 3. If everything looks good, delete test runs to keep database clean
if DEBUG:
    # Get track_id from test run
    track_ids = reader.to_df()['track_id'].unique()
    for tid in track_ids:
        print(f"Deleting test track_id: {tid}")
        # Delete test data (CLI method)
        subprocess.run(['xt', 'delete', 'benchmark.db', tid], check=True)
        # Or use Python API (if available)
        # tracker.delete(track_id=tid)

    print("✅ Test runs deleted. Ready to run full benchmark.")
    print("   Set DEBUG = False and run again.")
```

**Why this matters:**
- **Catch bugs early** - Fix issues with 10 items, not 10,000
- **Save time** - Don't wait hours to discover a parameter was wrong
- **Clean database** - Delete test runs so they don't pollute analysis
- **Iterate quickly** - Test → debug → delete → repeat until perfect

**Deleting test runs:**
```bash
# CLI: Delete by track_id
xt delete benchmark.db ancient-falcon-1234

# Delete multiple track_ids
for tid in test-id-1 test-id-2; do
    xt delete benchmark.db $tid
done

# Or delete all data from specific table (nuclear option)
xt sql benchmark.db "DELETE FROM db.predictions WHERE track_id = 'ancient-falcon-1234'"
```

### Full Benchmark

Once validated with subset, run the full benchmark using **both tables**:

```python
from xetrack import Tracker, Reader

# Run predictions for all parameter combinations
params_grid = [
    BenchmarkParams(model_name='bert-base', embedding_dim=768),
    BenchmarkParams(model_name='roberta-base', embedding_dim=768),
    BenchmarkParams(model_name='distilbert', embedding_dim=768),
]

for params in params_grid:
    print(f"Running: {params.model_name}...")

    # Tracker for individual predictions
    predictions_tracker = Tracker(
        db='benchmark.db',
        engine='duckdb',
        cache='cache_dir',
        table='predictions',  # Individual results
        params={'experiment_id': f'exp-{params.model_name}'}
    )

    # Tracker for aggregated metrics
    metrics_tracker = Tracker(
        db='benchmark.db',
        engine='duckdb',
        table='metrics',  # Aggregated results
        params={'experiment_id': f'exp-{params.model_name}'}
    )

    # Run predictions
    results = []
    for item in dataset:
        result = predictions_tracker.track(run_single_prediction, args=[item, params])
        results.append(result)

    # Calculate and log aggregated metrics
    successful = [r for r in results if r.get('error') is None]
    if successful:
        accuracy = sum(1 for r in successful if r['prediction'] == r['ground_truth']) / len(successful)
        avg_latency = sum(r['latency'] for r in successful) / len(successful)

        metrics_tracker.log({
            'model_name': params.model_name,
            'embedding_dim': params.embedding_dim,
            'accuracy': accuracy,
            'avg_latency': avg_latency,
            'n_predictions': len(dataset),
            'n_successful': len(successful),
            'n_failed': len(results) - len(successful)
        })

    accuracy_str = f"{accuracy:.4f}" if successful else "N/A (all failed)"
    print(f"  Completed {len(dataset)} predictions - Accuracy: {accuracy_str}")
```

**Why two tables?**
- **Predictions table**: Detailed data for segmentation ("which examples did Model A get wrong?")
- **Metrics table**: Quick comparison ("which model is best overall?")

**Rerun Safety:** If script crashes, restart it! Cached results prevent re-execution.

---
