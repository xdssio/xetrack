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
   - **MUST READ**: `/path/to/xetrack/README.md` - Full API documentation
   - **MUST READ**: `/path/to/xetrack/examples/README.md` - Examples guide
   - **Run example**: `python examples/01_quickstart.py` to verify installation

4. **Understand core concepts:**
   - `Tracker(db, params=dict, engine='sqlite|duckdb', table='events')` - Main tracking interface
   - `tracker.log(dict)` - Log arbitrary data
   - `tracker.track(func, args, kwargs)` - Track function execution with auto-unpacking
   - `Reader(db, engine='sqlite|duckdb', table='events')` - Read tracked data
   - `xt` CLI - Command-line interface for queries

**⚠️ IMPORTANT:** xetrack is not a common package. DO NOT hallucinate APIs. Always reference README/examples for correct usage patterns.

## Workflow

Follow this sequential workflow with users:

### 1. Understand Goals → Design Experiment
### 2. Build Single-Execution Function → Validate
### 3. Add Caching → Test Reproducibility
### 4. Parallelize (if needed) → Run
### 5. Validate Results → Analyze

---

## Critical: Development Phase vs Experiment Phase

**IMPORTANT:** Benchmarking has two distinct phases. Do NOT mix them!

### 🔧 Development Phase (Testing & Iteration)

**Purpose:** Build and test your benchmark pipeline until it's bug-free.

**Activities:**
- Write unit tests for your prediction function
- Test with small subsets (10-100 samples)
- Iterate rapidly on code
- Delete test runs frequently
- Use separate track_id or table for testing (e.g., `test_predictions`)

**Testing requirements:**
```python
import pytest

def test_prediction_function():
    """Unit test: ensure prediction function works correctly."""
    params = ModelParams(model='test', threshold=0.5)
    item = {'id': 1, 'x': 0.7, 'label': True}

    result = predict(item, params)

    # Verify output structure
    assert 'input_id' in result
    assert 'prediction' in result
    assert 'ground_truth' in result
    assert isinstance(result['prediction'], bool)

def test_error_handling():
    """
    CRITICAL TEST: Ensure errors are caught and logged, not raised.

    Your prediction function MUST handle errors gracefully:
    - Catch exceptions
    - Return error info in result dict
    - Don't break the entire experiment loop
    """
    params = ModelParams(model='test', threshold=0.5)
    bad_item = {'id': 1}  # Missing required fields

    # This should NOT raise an exception
    result = predict(bad_item, params)

    # Error should be captured in result
    assert 'error' in result
    assert result['error'] is not None
    assert 'error_type' in result  # Categorize errors

    # Other fields should still be present (even if null)
    assert 'input_id' in result
    assert 'prediction' in result  # Should be None or default value

def test_caching():
    """Test that same input gives cache hit."""
    params = ModelParams(model='test', threshold=0.5)
    item = {'id': 1, 'x': 0.7, 'label': True}

    # First call
    result1 = tracker.track(predict, args=[item, params])

    # Second call (should be cached)
    result2 = tracker.track(predict, args=[item, params])

    # Both should have same result
    assert result1['prediction'] == result2['prediction']
```

**Development workflow:**
```python
# DEV MODE: Use separate testing database or table
dev_tracker = Tracker(
    db='dev_benchmark.db',  # Or use table='test_predictions'
    engine='sqlite',
    cache='dev_cache',
    table='test_predictions',
    params={'phase': 'development'}
)

# Test with small subset
test_dataset = dataset[:10]

# Run tests
pytest -v tests/

# Run on subset
for item in test_dataset:
    result = dev_tracker.track(predict, args=[item, params])

# ========================================
# CRITICAL: INSPECT THE DATA
# ========================================
inspect_dev_data('dev_benchmark.db', 'test_predictions')

# If bugs found → fix → delete test runs → test again
subprocess.run(['xt', 'delete', 'dev_benchmark.db', test_track_id])
```

### Data Inspection During Development

**Always inspect your data before moving to production!**

```python
def inspect_dev_data(db_path, table_name, n_samples=10):
    """
    Peek at tracked data to catch issues during development.

    Checks for:
    - Missing values (NULLs)
    - Nonsensical values (negative probabilities, out-of-range)
    - Data type issues
    - Empty strings or zero values where unexpected
    """
    from xetrack import Reader
    import pandas as pd

    print("\n" + "="*70)
    print(f"📊 DATA INSPECTION: {table_name}")
    print("="*70 + "\n")

    # Read data
    reader = Reader(db=db_path, table=table_name)
    df = reader.to_df()

    if len(df) == 0:
        print("❌ No data found! Did the tracking work?")
        return False

    print(f"✅ Found {len(df)} rows\n")

    issues_found = []

    # ========================================
    # 1. SHOW SAMPLE DATA
    # ========================================
    print(f"1️⃣  SAMPLE DATA (first {n_samples} rows)")
    print("-" * 70)

    # Show key columns (exclude internal xetrack columns for clarity)
    display_cols = [c for c in df.columns
                    if c not in ['timestamp', 'track_id', 'function_name', 'function_time']]

    print(df[display_cols].head(n_samples).to_string())
    print()

    # ========================================
    # 2. CHECK FOR MISSING VALUES
    # ========================================
    print("2️⃣  MISSING VALUES CHECK")
    print("-" * 70)

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)

    has_missing = False
    for col in df.columns:
        if missing[col] > 0:
            has_missing = True
            pct = missing_pct[col]
            symbol = "⚠️" if pct > 50 else "❌"
            print(f"{symbol} {col}: {missing[col]} missing ({pct}%)")

            if col.startswith('params_') or col in ['prediction', 'ground_truth', 'raw_response']:
                issues_found.append(f"Missing values in critical field: {col}")

    if not has_missing:
        print("✅ No missing values")
    print()

    # ========================================
    # 3. CHECK FOR NONSENSICAL VALUES
    # ========================================
    print("3️⃣  DATA VALIDATION CHECK")
    print("-" * 70)

    checks_run = 0

    # Check probabilities (should be 0-1)
    prob_cols = [c for c in df.columns if 'prob' in c.lower() or 'confidence' in c.lower()]
    for col in prob_cols:
        checks_run += 1
        if df[col].dtype in ['float64', 'float32']:
            invalid = ((df[col] < 0) | (df[col] > 1)).sum()
            if invalid > 0:
                print(f"❌ {col}: {invalid} values outside [0, 1] range")
                issues_found.append(f"{col} has probabilities outside [0, 1]")
            else:
                print(f"✅ {col}: all values in [0, 1]")

    # Check for negative values in fields that shouldn't be negative
    non_negative_keywords = ['latency', 'time', 'duration', 'cost', 'count', 'length', 'tokens']
    non_negative_cols = [c for c in df.columns
                        if any(kw in c.lower() for kw in non_negative_keywords)]

    for col in non_negative_cols:
        checks_run += 1
        if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            negative = (df[col] < 0).sum()
            if negative > 0:
                print(f"❌ {col}: {negative} negative values (should be >= 0)")
                issues_found.append(f"{col} has negative values")
            else:
                print(f"✅ {col}: all values >= 0")

    # Check for empty strings in important text fields
    text_cols = [c for c in df.columns if 'response' in c.lower() or 'text' in c.lower()]
    for col in text_cols:
        checks_run += 1
        if df[col].dtype == 'object':
            empty = (df[col].str.strip() == '').sum()
            null = df[col].isnull().sum()
            if empty > 0 or null > 0:
                print(f"⚠️ {col}: {empty} empty strings, {null} nulls")
                if col == 'raw_response':
                    issues_found.append("raw_response has empty values - not saving model output!")

    # Check for all-zero or all-same values
    for col in df.columns:
        checks_run += 1
        if col not in ['timestamp', 'track_id']:
            n_unique = df[col].nunique()
            if n_unique == 1:
                value = df[col].iloc[0]
                print(f"⚠️ {col}: ALL values are the same ({value})")
                if col.startswith('params_'):
                    # This might be OK if testing one config
                    pass
                else:
                    issues_found.append(f"{col} has no variation (all same value)")

    if checks_run == 0:
        print("ℹ️ No numeric fields to validate")
    print()

    # ========================================
    # 4. CHECK DATA TYPES
    # ========================================
    print("4️⃣  DATA TYPES")
    print("-" * 70)

    for col in df.columns:
        dtype = df[col].dtype

        # Flag if numeric field stored as string
        if dtype == 'object' and col in ['accuracy', 'loss', 'latency', 'cost', 'confidence']:
            print(f"⚠️ {col}: stored as STRING but should be NUMERIC")
            issues_found.append(f"{col} stored as string instead of numeric")

    # Show data types for reference
    print("\nData types summary:")
    print(df.dtypes.value_counts().to_string())
    print()

    # ========================================
    # 5. SUMMARY
    # ========================================
    print("="*70)
    if issues_found:
        print("❌ ISSUES FOUND - FIX BEFORE MOVING TO PRODUCTION")
        print("="*70)
        for i, issue in enumerate(issues_found, 1):
            print(f"  {i}. {issue}")
        print()
        print("Common fixes:")
        print("  • Missing values: Check if function returns all expected fields")
        print("  • Wrong ranges: Check calculation logic (probabilities, times, etc.)")
        print("  • Empty raw_response: Ensure model output is being captured")
        print("  • All same values: Check if parameter is actually varying")
        print("  • Wrong data types: Check return types from function")
        print()
        return False
    else:
        print("✅ DATA LOOKS GOOD - NO ISSUES FOUND")
        print("="*70)
        print()
        return True

# Example usage in dev workflow:
for item in test_dataset:
    result = dev_tracker.track(predict, args=[item, params])

# Inspect before moving to production
if not inspect_dev_data('dev_benchmark.db', 'test_predictions'):
    print("\n⚠️ Fix data issues before moving to production!")
    print("1. Fix your predict() function")
    print("2. Delete test data: xt delete dev_benchmark.db <track_id>")
    print("3. Run tests again")
    print("4. Re-inspect until clean")
    exit(1)

# Validate error handling strategy
if not validate_error_handling('dev_benchmark.db', 'test_predictions'):
    print("\n⚠️ Error handling not validated!")
    exit(1)
```

### Error Handling Validation

**CRITICAL:** Your prediction function MUST handle errors gracefully - errors should be tracked, not break the experiment.

```python
def validate_error_handling(db_path, table_name):
    """
    Enforce error handling validation during development.

    Checks:
    1. Does the schema have 'error' and 'error_type' fields?
    2. Are there any test errors in the data?
    3. Does user understand the error handling strategy?
    """
    from xetrack import Reader

    print("\n" + "="*70)
    print("⚠️  ERROR HANDLING VALIDATION")
    print("="*70 + "\n")

    # Read data
    reader = Reader(db=db_path, table=table_name)
    df = reader.to_df()

    if len(df) == 0:
        print("❌ No data to validate")
        return False

    # Check if error fields exist
    has_error_field = 'error' in df.columns
    has_error_type = 'error_type' in df.columns

    print("1️⃣  ERROR FIELD CHECK")
    print("-" * 70)

    if not has_error_field:
        print("❌ CRITICAL: No 'error' field in data!")
        print("   Your prediction function MUST return an 'error' field.")
        print()
        print("   Example:")
        print("   def predict(item, params):")
        print("       try:")
        print("           # ... your prediction logic ...")
        print("           return {")
        print("               'prediction': result,")
        print("               'error': None,  # ← No error")
        print("               'error_type': None")
        print("           }")
        print("       except Exception as e:")
        print("           return {")
        print("               'prediction': None,  # ← Failed")
        print("               'error': str(e),     # ← Error message")
        print("               'error_type': type(e).__name__  # ← Error type")
        print("           }")
        print()
        return False

    print("✅ 'error' field exists\n")

    # Check for errors in data
    print("2️⃣  ERROR FREQUENCY")
    print("-" * 70)

    error_count = df['error'].notna().sum()
    error_rate = error_count / len(df) * 100

    if error_count > 0:
        print(f"⚠️  Found {error_count} errors ({error_rate:.1f}%)")

        if has_error_type:
            print("\n   Error breakdown:")
            error_types = df[df['error'].notna()]['error_type'].value_counts()
            for error_type, count in error_types.items():
                print(f"   - {error_type}: {count}")

        print("\n   This is GOOD - your error handling is working!")
        print("   Errors are being caught and tracked, not breaking the loop.")
    else:
        print("ℹ️  No errors found in test data")
        print("   This is OK, but make sure you've tested error cases.")

    print()

    # Force user decision on error handling strategy
    print("3️⃣  ERROR HANDLING STRATEGY")
    print("-" * 70)
    print("You MUST decide how to handle errors in experiments:\n")

    print("Options:")
    print("  [1] TRACK and CONTINUE (Recommended)")
    print("      - Errors are logged with 'error' field")
    print("      - Experiment continues with remaining items")
    print("      - Analyze errors after completion")
    print("      - Pro: Get results even with some failures")
    print("      - Con: May not notice critical issues immediately")
    print()
    print("  [2] TRACK and SKIP (Partial results)")
    print("      - Errors are logged")
    print("      - Failed items don't contribute to metrics")
    print("      - Still get results from successful items")
    print()
    print("  [3] STOP on first error (Debug mode)")
    print("      - Useful during development")
    print("      - NOT recommended for production experiments")
    print("      - Wastes time on transient errors (API timeouts, etc.)")
    print()

    choice = input("Your error handling strategy [1/2/3]: ").strip()

    if choice == '1':
        print("\n✅ TRACK and CONTINUE strategy selected")
        print("   Implementation:")
        print("   - Wrap predict() in try/except")
        print("   - Return {'error': str(e), 'error_type': type(e).__name__}")
        print("   - Filter errors during analysis: df[df['error'].isna()]")
        print()
        return True

    elif choice == '2':
        print("\n✅ TRACK and SKIP strategy selected")
        print("   Implementation:")
        print("   - Wrap predict() in try/except")
        print("   - Return {'error': str(e), ...}")
        print("   - In metrics calculation: results = [r for r in results if r['error'] is None]")
        print()
        return True

    elif choice == '3':
        print("\n⚠️  STOP on error strategy")
        print("   This is only for development/debugging.")
        print("   For production experiments, use strategy [1] or [2].")
        print()
        return False

    else:
        print("\n❌ Invalid choice. You MUST select an error handling strategy.")
        return False

# Example implementation of strategy #1 (TRACK and CONTINUE):
def predict_with_error_handling(item, params):
    """
    Recommended pattern: Track errors but don't break the loop.
    """
    try:
        # Your actual prediction logic
        if 'text' not in item:
            raise ValueError("Missing 'text' field")

        prediction = model.predict(item['text'], params)

        return {
            'input_id': item['id'],
            'prediction': prediction,
            'raw_response': model.last_response,
            'confidence': model.confidence,
            'error': None,           # ← No error
            'error_type': None
        }

    except Exception as e:
        # Error occurred - track it but DON'T raise
        return {
            'input_id': item.get('id'),  # Might not exist
            'prediction': None,          # ← Failed
            'raw_response': None,
            'confidence': None,
            'error': str(e),             # ← Track error message
            'error_type': type(e).__name__  # ← Track error type
        }

# During analysis, you can:
# 1. Filter out errors: successful = df[df['error'].isna()]
# 2. Analyze errors: errors = df[df['error'].notna()]
# 3. Calculate error rate: error_rate = len(errors) / len(df)
# 4. Group by error type: errors.groupby('error_type').size()
```

**Why this matters:**

1. **Don't break experiments** - One API timeout shouldn't waste 3 hours of compute
2. **Errors are data** - Track what fails and why
3. **Post-hoc analysis** - Decide later if errors are acceptable
4. **Robustness** - Production systems have transient failures

**Common error types to handle:**
```python
# API errors
except requests.exceptions.Timeout:
    error_type = 'API_TIMEOUT'

except openai.RateLimitError:
    error_type = 'RATE_LIMIT'

# Data errors
except KeyError as e:
    error_type = 'MISSING_FIELD'

except ValueError as e:
    error_type = 'INVALID_VALUE'

# Model errors
except Exception as e:  # Catch-all
    error_type = 'UNKNOWN'
```

**Exit criteria (ready for experiment phase):**
- ✅ All unit tests pass
- ✅ Tested on small subset (10-100 samples) successfully
- ✅ No errors in test runs
- ✅ Cache working correctly
- ✅ Schema validated (no parameter renames)
- ✅ Output format correct and consistent
- ✅ Code committed to git

### 🔬 Experiment Phase (Production Runs)

**Purpose:** Run full, reproducible experiments for analysis.

**Rules:**
- ⛔ **NO CODE CHANGES during experiments** - If you need to change code, go back to dev phase
- ✅ Run on full dataset
- ✅ Track everything (code commit, data commit, timestamps)
- ✅ Use production database/table (e.g., `predictions`, `metrics`)
- ✅ Data must be committed (DVC)
- ✅ Create git tags for each experiment

**Experiment workflow:**
```python
# PRODUCTION MODE: Full dataset, everything tracked
prod_tracker = Tracker(
    db='benchmark.db',
    engine='duckdb',
    cache='cache',
    table='predictions',
    params={
        'experiment_version': 'e0.0.1',
        'code_commit': get_git_hash()[:7],
        'data_commit': get_data_version()[:7]
    }
)

# Pre-run validation
assert validate_schema_before_experiment('benchmark.db', 'predictions', ModelParams)
assert check_data_committed()  # Optional but recommended

# Run full benchmark
for item in full_dataset:  # All data
    result = prod_tracker.track(predict, args=[item, params])

# Tag experiment
subprocess.run(['git', 'tag', '-a', 'e0.0.1', '-m', 'baseline model'])
```

### ⚠️ If You Need to Change Code During Experiments

**STOP! Go back to development phase:**

```python
# ❌ WRONG: Making code changes during experiment phase
# This breaks reproducibility!

# ✅ RIGHT: Go back to development phase
print("⚠️  Code change needed. Returning to development phase.")

# 1. Switch to dev database/table
dev_tracker = Tracker(db='dev_benchmark.db', ...)

# 2. Make code changes
# ... edit code ...

# 3. Test changes on small subset
# ... run tests ...

# 4. When tests pass, commit code
subprocess.run(['git', 'add', '.'])
subprocess.run(['git', 'commit', '-m', 'fix: correct prediction logic'])

# 5. Delete any partial experiment runs
subprocess.run(['xt', 'delete', 'benchmark.db', partial_track_id])

# 6. Restart experiment phase with new code
new_experiment_version = increment_tag('e0.0.1')  # e0.0.2
prod_tracker = Tracker(db='benchmark.db', params={'experiment_version': new_experiment_version})
```

### Phase Transition Checklist

**Before moving from DEV → EXPERIMENT:**

```python
def ready_for_production():
    """Checklist: Are we ready to leave development phase?"""
    checks = []

    # 1. Tests pass
    result = subprocess.run(['pytest', '-v'], capture_output=True)
    checks.append(('All tests pass', result.returncode == 0))

    # 2. Schema validated
    schema_ok = validate_schema_before_experiment('benchmark.db', 'predictions', ModelParams)
    checks.append(('Schema validated', schema_ok))

    # 3. Code committed
    result = subprocess.run(['git', 'diff', '--quiet'], capture_output=True)
    checks.append(('Code committed', result.returncode == 0))

    # 4. Data committed (if using DVC)
    data_ok = check_data_committed()
    checks.append(('Data committed', data_ok))

    # Print checklist
    print("\n📋 Production Readiness Checklist:")
    all_pass = True
    for check_name, passed in checks:
        symbol = '✅' if passed else '❌'
        print(f"   {symbol} {check_name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n✅ READY FOR PRODUCTION EXPERIMENTS")
    else:
        print("\n❌ NOT READY - Fix issues above")

    return all_pass

# Run before starting experiments
if not ready_for_production():
    print("Staying in development phase.")
    exit(1)
```

**Summary:**
- 🔧 **Dev phase** = Fast iteration, testing, small subsets, deletable runs
- 🔬 **Experiment phase** = No code changes, full data, reproducible, git tagged
- ⚠️ **Code change during experiment** = Go back to dev phase

---

## Phase 0: Ideation - Plan What to Track

**Do this BEFORE writing any code.** You can't analyze what you didn't collect.

### Three Critical Questions

#### 1. What questions do you want to answer?

**This determines what data you need to collect.**

```python
# Example questions:
questions = [
    "Which model is most accurate?",
    "How does performance vary by input length?",
    "What's the cost vs accuracy tradeoff?",
    "Where do models fail? (failure modes)",
    "How does domain affect results? (medical vs legal vs casual text)",
    "Is latency acceptable for production?",
    "Does performance degrade with batch size?"
]

# Each question implies data requirements:
# "Most accurate?" → Need: model_name, accuracy, ground_truth
# "Vary by input length?" → Need: input_length or prompt_length
# "Cost vs accuracy?" → Need: cost_per_call, total_tokens, accuracy
# "Failure modes?" → Need: error_type, error_message, raw_input
# "Domain affect?" → Need: domain label (medical/legal/casual)
# "Latency?" → Need: inference_time, batch_size
```

#### 2. What segmentations do you want to perform?

**This determines what parameters you should track.**

If you don't log these fields now, no amount of clever analysis will save you later.

```python
# Common segmentation dimensions:

# INPUT CHARACTERISTICS
'input_length'      # Prompt length in tokens/chars
'domain'            # medical, legal, casual, technical
'difficulty'        # easy, medium, hard
'language'          # en, es, fr, zh
'input_type'        # question, statement, command

# MODEL PARAMETERS
'model_name'        # gpt-4, claude-3, llama-2
'temperature'       # 0.0, 0.7, 1.0
'max_tokens'        # 100, 500, 2000
'top_p'             # nucleus sampling parameter
'system_prompt'     # hash or short identifier

# OPERATIONAL
'batch_size'        # 1, 8, 32
'cache_hit'         # true/false (auto-tracked by xetrack)
'retry_count'       # number of retries before success
'error_type'        # timeout, rate_limit, invalid_output

# BUSINESS/COST
'cost_usd'          # dollar cost per call
'total_tokens'      # input + output tokens
'region'            # us-east, eu-west (API region)

# QUALITY
'confidence_score'  # model's confidence
'failure_mode'      # hallucination, refusal, incomplete
'human_rating'      # 1-5 stars (if available)
```

**Anti-pattern:** Only tracking model_name and accuracy:
```python
# ❌ BAD: Can only compare models, nothing else
params = {'model': 'gpt-4'}
result = {'accuracy': 0.85}

# Later you want to know: "How does GPT-4 perform on long inputs?"
# Answer: Can't tell - didn't track input_length!
```

**Good pattern:** Track dimensions you might want to slice by:
```python
# ✅ GOOD: Can answer many questions later
params = {
    'model': 'gpt-4',
    'temperature': 0.7,
    'domain': 'medical',
    'input_length_bucket': '500-1000'  # Bucketed for easier grouping
}
result = {
    'accuracy': 0.85,
    'latency_ms': 1234,
    'cost_usd': 0.002,
    'confidence': 0.92,
    'error': None
}

# Now you CAN answer: "How does GPT-4 perform on long medical inputs?"
```

#### 3. What else can you collect without slowing down development?

**xetrack automatically tracks:**
- ✅ `timestamp` - When the execution happened
- ✅ `track_id` - Unique identifier for the run
- ✅ `function_time` - How long the function took
- ✅ CPU, memory, network (if `log_system_params=True`)

**You should also track:**
```python
import subprocess
import platform

def get_environment_metadata():
    """Collect metadata that's cheap but valuable later."""
    return {
        # Code versioning
        'code_commit': subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']
        ).decode().strip(),
        'git_branch': subprocess.check_output(
            ['git', 'branch', '--show-current']
        ).decode().strip(),
        'data_version': subprocess.check_output(
            ['git', 'log', '-n', '1', '--pretty=format:%h', '--', 'data.dvc']
        ).decode().strip() if os.path.exists('data.dvc') else None,

        # Environment
        'python_version': platform.python_version(),
        'os': platform.system(),
        'hostname': platform.node(),

        # Model versioning (for LLMs)
        'model_version': 'gpt-4-0613',  # API version identifier
        'sdk_version': openai.__version__,  # Library version

        # Hardware (if relevant)
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'num_gpus': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

# Include in tracker params
tracker = Tracker(
    db='benchmark.db',
    params=get_environment_metadata()
)
```

### 💡 Key Insight: Storage is Cheap, Regret is Expensive

> **Future you will appreciate past you's paranoia.**

**Why over-collect:**
- Adding fields later means old data lacks them (holes in analysis)
- Storage costs pennies, re-running experiments costs hours/dollars
- You don't know what questions you'll ask in 6 months
- Debugging mysterious issues requires context you forgot to log

### ⚠️ CRITICAL: Always Save Raw Responses

> **The data you didn't save is the question you'll care about most.**

**Never benchmark on post-processed outputs alone.**

**Anti-pattern (loses information):**
```python
# ❌ BAD: Only saving processed result
def evaluate_model(input_text, model):
    response = model.generate(input_text)  # Raw response from model

    # Parse to get answer
    answer = extract_answer(response)  # Parsing happens HERE
    is_correct = (answer == ground_truth)

    return {'correct': is_correct}  # ← Raw response is LOST forever!

# Later: "Was the model close? What did it actually say?"
# Answer: Can't tell - raw response was discarded!
```

**Good pattern (preserves raw data):**
```python
# ✅ GOOD: Save EVERYTHING, parse later
def evaluate_model(input_text, model):
    # Execute and collect raw data
    start = time.time()
    response = model.generate(input_text)
    latency = time.time() - start

    return {
        # RAW DATA (cannot be reconstructed)
        'raw_response': response,              # Actual model output
        'raw_logits': model.get_logits(),      # If available
        'raw_probabilities': model.get_probs(), # If available

        # METADATA (cheap to collect)
        'input_text': input_text,
        'input_length': len(input_text),
        'output_length': len(response),
        'latency_ms': latency * 1000,
        'tokens_used': model.last_token_count,
        'model_version': model.version,

        # ERRORS (if any)
        'error': None,
        'error_type': None,
        'retry_count': 0,

        # Ground truth for comparison
        'ground_truth': ground_truth,

        # NOTE: Do NOT parse here! Parse during analysis phase
        # This preserves ability to change parsing logic later
    }

# Later during analysis, parse THEN:
results_df = Reader('benchmark.db', table='predictions').to_df()

# Try different parsing strategies
results_df['answer_v1'] = results_df['raw_response'].apply(extract_answer_v1)
results_df['answer_v2'] = results_df['raw_response'].apply(extract_answer_v2)
results_df['answer_v3'] = results_df['raw_response'].apply(extract_answer_v3)

# Compare which parser works best
accuracy_v1 = (results_df['answer_v1'] == results_df['ground_truth']).mean()
accuracy_v2 = (results_df['answer_v2'] == results_df['ground_truth']).mean()
accuracy_v3 = (results_df['answer_v3'] == results_df['ground_truth']).mean()
```

**Why this matters:**

1. **Post-processing should happen AFTER execution, not during**
   - Parsing is analysis, not data collection
   - You might want to try different parsing strategies
   - Parsing logic might have bugs (common!)

2. **A single parsing decision made early removes entire classes of future analysis**
   ```python
   # Early decision: "Save only if answer is non-empty"
   if answer.strip():
       save(answer)

   # Later: "Actually, empty answers are acceptable for some inputs"
   # Can't recover this data - you filtered it out!
   ```

3. **You cannot reconstruct what the model actually said**
   - Model APIs change versions
   - You can't re-query historical model versions
   - Re-running is expensive (time + API costs)

4. **Raw data enables unexpected analysis**
   ```python
   # Examples of analysis you couldn't do without raw responses:

   # - Did model refuse to answer? (Check for "I cannot...")
   # - Was answer confident or hedged? (Check for "maybe", "perhaps")
   # - Did model hallucinate? (Check output vs ground truth)
   # - Length of explanation? (Count words/tokens)
   # - Did model follow format? (Regex match expected structure)
   # - Failure mode analysis (Cluster similar bad responses)
   ```

**What to save for different benchmark types:**

```python
# LLM Benchmarking
result = {
    'raw_response': response_text,      # The actual text generated
    'raw_logprobs': logprobs,           # Token probabilities if available
    'finish_reason': 'stop',            # stop, length, content_filter
    'prompt': full_prompt,              # Exact prompt sent
    'completion_tokens': 234,
    'prompt_tokens': 156,
    'total_tokens': 390,
    'model_version': 'gpt-4-0613',
    'system_prompt': system_prompt_text,
}

# Classification Model
result = {
    'raw_probabilities': [0.1, 0.7, 0.2],  # All class probabilities
    'predicted_class': 1,                   # Argmax (can derive)
    'predicted_prob': 0.7,                  # Max prob (can derive)
    'logits': [-2.3, 0.8, -1.6],           # Pre-softmax
    'ground_truth_class': 1,
}

# Embedding Model
result = {
    'raw_embedding': embedding_vector,      # Full vector (e.g., 768 dims)
    'embedding_norm': np.linalg.norm(embedding_vector),
    'embedding_hash': hash(embedding_vector.tobytes()),  # For dedup
    # Don't just save: 'cosine_similarity': 0.85  # Can compute from vectors!
}

# Regression Model
result = {
    'predicted_value': 42.3,
    'prediction_std': 2.1,           # Uncertainty if available
    'ground_truth_value': 40.0,
    'raw_features': feature_dict,    # Input features used
}
```

**Summary:**
- 💾 **Save raw responses** - You can always parse later
- ❌ **Don't parse during execution** - Post-processing is analysis
- 🔮 **You can't predict future questions** - Raw data enables unexpected analysis
- 💰 **Storage is cheap** - Regret is expensive

**Example regret:**
```python
# Week 1: "We only care about accuracy"
result = {'accuracy': 0.85}

# Week 6: "Why did accuracy drop from 0.85 to 0.70?"
# Possible causes:
# - Input distribution changed? (didn't track domain/length)
# - API changed? (didn't track model_version)
# - Data changed? (didn't track data_commit)
# - Bug introduced? (didn't track code_commit)
#
# Answer: ¯\_(ツ)_/¯  Can't tell - didn't log enough!
```

### Ideation Checklist

Before writing code, document:

```python
# Save this in experiment_plan.md or docstring

"""
BENCHMARK IDEATION

Questions to Answer:
1. Which embedding model is most accurate on medical text?
2. How does accuracy vary by document length?
3. What's the cost vs accuracy tradeoff?
4. What are common failure modes?

Segmentations Planned:
- model_name: ['bert-base', 'roberta', 'biomed-bert']
- domain: ['medical', 'legal', 'casual']
- doc_length_bucket: ['0-500', '500-1000', '1000+']
- difficulty: ['easy', 'medium', 'hard']

Tracked Fields:
- Input: doc_id, text_length, domain, difficulty
- Params: model_name, embedding_dim, max_length
- Output: prediction, ground_truth, confidence, embedding_hash
- Cost: inference_time_ms, cost_usd
- Context: code_commit, data_commit, timestamp
- Errors: error_type, error_message (if any)

Storage Budget: ~10MB for 10K predictions (acceptable)
"""
```

**Output from Phase 0:**
- ✅ List of questions to answer
- ✅ List of segmentation dimensions to track
- ✅ Dataclass/schema with ALL fields planned
- ✅ Awareness of what you might regret NOT tracking

**Anti-patterns to avoid:**
- ❌ "We'll add more fields later if needed" (you'll forget)
- ❌ "We only need the basics" (until you don't)
- ❌ "Storage is expensive" (it's not, regret is)

---

## Phase 1: Understand Goals & Design Experiment

**Start from the end.** Before writing code, clarify:

### Ask the User:

1. **"What questions do you want to answer?"**
   - Example: "Is Model A better than Model B?" "Which prompt works best?" "Does my preprocessing improve accuracy?"

2. **"What comparisons or segmentations will you perform?"**
   - This determines what parameters to track
   - Examples: model type, hyperparameters, data subsets, prompt variations
   - **Common segmentations**: prompt length, domain, difficulty level, failure mode, cost bracket

   > *"Storage is cheap; regret is expensive. Future you will appreciate past you's paranoia."*

3. **"What metrics matter?"**
   - Accuracy, F1, latency, **cost** (especially for LLMs!), throughput?
   - Save raw outputs (probabilities, full LLM responses) for future re-analysis
   - **For LLMs**: Track token counts, cost per prediction, API latency

4. **"What data are you benchmarking on?"**
   - Size, format, location
   - Will it fit in memory? Need batching?

### Design Decisions:

Based on answers, recommend:

**Database Engine Decision Matrix:**

| Factor | SQLite | DuckDB |
|--------|--------|--------|
| **Multiprocessing** | ✅ Works | ❌ Database locks |
| **Single-process** | ✅ Works | ✅ Works |
| **Analytics queries** | ⚠️ Limited | ✅ Advanced SQL |
| **Large datasets** | ⚠️ Slower | ✅ Faster |
| **Schema flexibility** | ✅ Add columns later | ✅ Add columns later |
| **SQL table names** | `predictions` | `db.predictions` |
| **Default** | ✅ Yes | No |

**Note on table naming:** In Python code, always use simple table names like `table='predictions'`. The `db.` prefix is only needed when querying via DuckDB CLI after attaching the SQLite file: `ATTACH 'benchmark.db' AS db (TYPE sqlite);`

**Decision flowchart:**
```
Will you use multiprocessing?
├─ YES → Use SQLite (engine='sqlite')
└─ NO  → Use DuckDB (engine='duckdb') for better analytics
```

**Installation:**
```bash
pip install xetrack[duckdb]  # For DuckDB support
```

**Important:** Once you choose an engine, use it consistently throughout your benchmark!

**Table Organization (IMPORTANT - Two Tables Pattern):**

Always create **two separate tables**:

1. **Predictions table** (e.g., `predictions`, `training_steps`, `requests`)
   - Stores every single execution/prediction
   - One row per datapoint
   - Enables detailed segmentation analysis
   - Format: `(track_id, timestamp, input_id, model, params..., prediction, ground_truth, latency, error...)`

2. **Metrics table** (e.g., `metrics`, `final_metrics`, `throughput_summary`)
   - Stores aggregated results per experiment configuration
   - One row per parameter combination
   - Enables quick model comparison
   - Format: `(track_id, timestamp, model, params..., accuracy, avg_latency, total_cost, n_samples...)`

**Naming suggestions:**
- Predictions: `predictions`, `inferences`, `training_steps`, `eval_checkpoints`, `requests`
- Metrics: `metrics`, `summary`, `final_metrics`, `experiment_results`

Users can customize table names, but this two-table pattern is recommended.

**Parameter Tracking:**
- Use frozen dataclasses for all experiment parameters (enables caching)
- Track git commit hash if reproducibility is critical
- Track data version (use DVC commit hash: `git log -n 1 --pretty=format:%H -- data.dvc`)
- Track timestamps, model versions, hardware specs (xetrack does this automatically)
- **Experiment naming:** xetrack uses coolname (e.g., `purple-mountain-4392`) - human-readable and unique

### Output from Phase 1:

A clear specification:
```python
# Example output
"""
Goal: Compare 3 embedding models on text classification
Data: 1000 labeled examples
Metrics: Accuracy, F1, inference latency
Params to track: model_name, embedding_dim, batch_size
Database: benchmark.db with DuckDB engine
Tables: predictions (individual), experiments (aggregated)
"""
```

---

## Phase 2: Build Single-Execution Function

**Critical principle: Every datapoint executes exactly once.**

### Single-Execution Pattern:

```python
from dataclasses import dataclass
from xetrack import Tracker

@dataclass(frozen=True, slots=True)  # frozen=True makes it hashable for caching
class BenchmarkParams:
    """All parameters that affect the result."""
    model_name: str
    embedding_dim: int
    temperature: float = 0.0
    batch_size: int = 32
    # Use immutable types only: no lists/dicts, use tuples/frozensets

def run_single_prediction(input_data: dict, params: BenchmarkParams) -> dict:
    """
    Single-execution function: stateless, thread-safe, cacheable.

    Returns everything you might need later, including:
    - prediction (the actual output)
    - raw_response (for re-analysis)
    - latency
    - error (if any)
    - metadata
    """
    import time
    start = time.time()

    try:
        # Your model inference here
        prediction = your_model.predict(input_data, params)
        raw_response = your_model.get_raw_output()
        error = None
    except Exception as e:
        prediction = None
        raw_response = None
        error = str(e)

    latency = time.time() - start

    return {
        'input_id': input_data['id'],
        'prediction': prediction,
        'raw_response': raw_response,  # ALWAYS save raw outputs
        'ground_truth': input_data.get('label'),
        'latency': latency,
        'error': error
    }
```

### Teach xetrack: Track with Frozen Dataclass

**Reference:** `examples/03_dataclass_unpacking.py` for complete example

```python
# Initialize tracker with DuckDB
# API Reference: Tracker(db, engine='sqlite|duckdb', table='events', params=dict, cache=str)
tracker = Tracker(
    db='benchmark.db',
    engine='duckdb',  # Recommended for benchmarks (or 'sqlite' for multiprocessing)
    table='predictions',
    params={'experiment_id': 'exp-001'}  # Groups this run
)

# Track execution - dataclass params are auto-unpacked!
# API Reference: tracker.track(function, args=list, kwargs=dict, params=dict)
params = BenchmarkParams(model_name='bert-base', embedding_dim=768)
result = tracker.track(
    run_single_prediction,
    args=[input_data, params]
)
# Tracked columns: input_id, prediction, latency, error, params_model_name, params_embedding_dim...
```

**Xetrack Feature: Automatic Dataclass Unpacking**
- Frozen dataclasses are automatically unpacked into individual columns
- `params.model_name` → `params_model_name` column
- Enables easy filtering: `SELECT * WHERE params_model_name = 'bert-base'`
- See `examples/03_dataclass_unpacking.py` for full documentation

**Validation Checkpoint:**
- ✓ Verify `Tracker()` parameters match README documentation
- ✓ Confirm `tracker.track()` usage matches `examples/02_track_functions.py`
- ✓ Check dataclass unpacking works as in `examples/03_dataclass_unpacking.py`

### Validation Checklist:

Before proceeding, validate:

- [ ] Function is **stateless** (no shared mutable state like global lists)
- [ ] Function is **deterministic** (same inputs → same outputs)
- [ ] Function returns **everything you might need** (including errors)
- [ ] Parameters are in a **frozen dataclass** (for caching)
- [ ] **Raw outputs are saved** (not just processed results)
- [ ] **Failures are captured** (error field, not silent failures)

**Run validation script:**
```bash
python scripts/validate_benchmark.py benchmark.db
```

---

## Phase 3: Add Caching

**Caching is not optimization—it's a correctness tool.** Prevents duplicate executions and wasted compute.

### Enable Caching:

**Reference:** `examples/05_function_caching.py` for complete caching guide

**Installation:** `pip install xetrack[cache]` (requires diskcache)

```python
# API Reference: Tracker(cache='directory_path')
tracker = Tracker(
    db='benchmark.db',
    engine='duckdb',
    cache='cache_dir',  # Enable disk-based caching
    table='predictions'
)

# First call: computes and caches
result1 = tracker.track(run_single_prediction, args=[input_data, params])

# Second call with same args: instant cache hit!
result2 = tracker.track(run_single_prediction, args=[input_data, params])
```

**Validation Checkpoint:**
- ✓ Installed `xetrack[cache]`?
- ✓ Cache parameter usage matches `examples/05_function_caching.py`?
- ✓ Verify cache directory is created
- ✓ Test cache hit by re-running same function call

**Caching Requirements:**
- All function arguments must be **hashable** (use frozen dataclasses)
- **Treat floats as hostile**—round or quantize before hashing (`round(temperature, 2)`)
- Lists/dicts break caching—use tuples/frozensets instead
- **"Cache you cannot observe is a liability"** - always verify caching works with validation scripts

**LLM-Specific Caching:**
For LLM benchmarks, consider specialized caching:
- **GPTCache**: Semantic similarity-based caching for LLM responses
- **LangChain caching**: Built-in cache backends for chat models
- These can cache even with slight prompt variations (semantic matching)
- Especially useful for expensive API calls (GPT-4, Claude)

**Cache Lineage Tracking:**
- xetrack tracks cache hits via the `cache` column
- Empty string `""` = computed (cache miss)
- track_id value = cache hit (references original execution)

### Teach xetrack CLI: Check Caching

**Reference:** README.md "CLI" section for all `xt` commands

```bash
# View first rows (xt head)
# API Reference: xt head <db> --n=<rows> --engine=<sqlite|duckdb>
xt head benchmark.db --n=10 --engine=duckdb
# Look for 'cache' column - track_id values indicate cache hits

# Execute SQL query (xt sql)
# API Reference: xt sql <db> "<query>" --engine=<sqlite|duckdb>
xt sql benchmark.db "SELECT input_id, params_model_name, cache FROM db.predictions LIMIT 10"

# Other useful commands:
xt tail benchmark.db --n=10  # View last rows
xt stats describe benchmark.db --columns=latency,accuracy  # Statistics
```

**Validation Checkpoint:**
- ✓ `xt` command available? (run `xt --help`)
- ✓ Commands match README CLI documentation?
- ✓ SQL syntax correct for engine (SQLite vs DuckDB)?

### Validation:

```bash
# Check cache effectiveness
python scripts/analyze_cache_hits.py benchmark.db predictions
```

Expected output:
```
Cache Analysis:
- Total executions: 1000
- Cache hits: 0 (0.0%) ← First run
- Cache misses: 1000 (100.0%)
- Unique parameter combinations: 3
```

---

## Common Pitfalls & Solutions

### Pitfall 1: Dataclass Unpacking Only Works with `.track()`

**Problem:**
```python
# This WON'T unpack dataclass params
tracker = Tracker(db='bench.db', params={'run_id': 'exp1'})
# Result: column named 'run_id', not 'params_run_id'
```

**Solution:**
- `Tracker(params={...})` stores keys as-is (no `params_` prefix)
- Dataclass unpacking ONLY happens when dataclass is a **function argument**:
```python
@dataclass(frozen=True)
class Config:
    model: str

def predict(data, config: Config):  # config is argument
    ...

tracker.track(predict, args=[data, Config(model='bert')])
# Result: columns include 'params_model', 'params_config_model'
```

### Pitfall 2: DuckDB + Multiprocessing = Database Locks

**Problem:**
```python
# This FAILS with "database is locked" error
with Pool(processes=4) as pool:
    results = pool.map(worker_func, data)  # Multiple processes write to DuckDB
```

**Solution:**
- **Use SQLite engine** for multiprocessing (handles concurrent writes better):
```python
tracker = Tracker(db='bench.db', engine='sqlite')  # NOT duckdb
```

- **Or use threading** instead of multiprocessing (single process, multiple threads)
- **Or batch results** in memory and write sequentially

### Pitfall 3: System Monitoring + Multiprocessing = AssertionError

**Problem:**
```python
# This FAILS: "daemonic processes are not allowed to have children"
tracker = Tracker(db='bench.db', log_system_params=True)
with Pool(processes=4) as pool:  # Fails!
    ...
```

**Solution:**
```python
# Disable system monitoring in multiprocessing contexts
tracker = Tracker(db='bench.db', log_system_params=False)
```

- System monitoring spawns child processes, incompatible with Pool workers
- Use system monitoring ONLY in single-process or threading contexts

### Pitfall 4: Model Objects Bloat Database

**Problem:**
```python
def train(X, y, params):
    model = RandomForest().fit(X, y)
    return {'model': model}  # Stores full model in database!

tracker.track(train, args=[X, y, params])
# Database grows to 100s of MB
```

**Solution:**
- **Save model hash only** (requires `pip install xetrack[assets]`):
```python
return {'model': model}  # xetrack saves as asset, stores hash in DB
```

- **Or don't return model** from tracked function (save separately):
```python
return {'model_path': 'models/model_001.pkl'}
# Save model outside of tracking
```

### Pitfall 5: Cache Column Missing

**Problem:**
Cache directory is created but `cache` column doesn't appear in database.

**Root causes:**
1. **Most likely:** DuckDB engine may not populate cache column (known limitation)
2. Cache feature requires `pip install xetrack[cache]` (diskcache)
3. xetrack version may not support cache tracking

**Solution:**
```python
# Try SQLite engine instead
tracker = Tracker(db='bench.db', engine='sqlite', cache='cache_dir')  # Not duckdb

# Verify cache column appears
df = Reader('bench.db', engine='sqlite').to_df()
print('cache' in df.columns)  # Should be True

# If still missing, verify diskcache installed:
# pip install xetrack[cache]
```

**Workaround:** Cache still works even if column is missing (results are cached), but you won't have lineage tracking.

### Pitfall 6: Float Parameters Break Caching

**Problem:**
```python
@dataclass(frozen=True)
class Config:
    learning_rate: float  # 0.0001 vs 0.00010000001 are different!

# These are treated as different configs even if functionally identical
tracker.track(train, args=[Config(learning_rate=1e-4)])
tracker.track(train, args=[Config(learning_rate=0.0001)])  # Cache miss!
```

**Solution:**
```python
@dataclass(frozen=True)
class Config:
    learning_rate: float

    def __post_init__(self):
        # Round floats for consistent hashing
        object.__setattr__(self, 'learning_rate', round(self.learning_rate, 6))
```

### Pitfall 7: Metrics Table Doesn't Have `params_*` Columns

**Problem:**
```python
metrics_tracker.log({
    'model': 'bert',
    'accuracy': 0.85
})
# Metrics table has 'model', NOT 'params_model'
```

**This is expected!** When using `.log()`, you manually control column names. To match predictions table format, manually include params:

```python
metrics_tracker.log({
    'model_type': params.model_type,  # Match your param names
    'regularization': params.regularization,
    'accuracy': accuracy
})
```

---

## Phase 4: Parallelize (Optional)

If benchmark is slow, parallelize after validating single-execution:

```python
from multiprocessing import Pool
from functools import partial

# Stateless function is safe to parallelize
def run_with_tracker(item, params):
    tracker = Tracker(db='benchmark.db', engine='duckdb',
                     cache='cache_dir', table='predictions')
    return tracker.track(run_single_prediction, args=[item, params])

# Parallel execution
params = BenchmarkParams(model_name='bert-base', embedding_dim=768)
with Pool(processes=4) as pool:
    results = pool.map(partial(run_with_tracker, params=params), data)
```

**Important:** Each process creates its own tracker. DuckDB/SQLite handle concurrent writes safely.

---

## Phase 5: Run Full Benchmark Loop

### Pre-Experiment Validation & Design Review

**Critical:** Before EVERY experiment, validate and review your design.

You might discover new things you want to collect or new aggregations for the metrics table.

```python
import subprocess
import sqlite3
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
        # Get last experiment's code commit
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT code_commit FROM {metrics_table} ORDER BY timestamp DESC LIMIT 1")
        result = cursor.fetchone()
        conn.close()

        if result:
            last_code_commit = result[0]
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

    # Show current schema
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Predictions table
    try:
        cursor.execute(f"PRAGMA table_info({predictions_table})")
        pred_columns = [row[1] for row in cursor.fetchall()]
        print(f"📊 Current predictions table columns ({len(pred_columns)}):")
        for col in sorted(pred_columns):
            print(f"   - {col}")
        print()
    except:
        print(f"ℹ️  Table '{predictions_table}' does not exist yet (first run)\n")
        pred_columns = []

    # Metrics table
    try:
        cursor.execute(f"PRAGMA table_info({metrics_table})")
        metrics_columns = [row[1] for row in cursor.fetchall()]
        print(f"📈 Current metrics table columns ({len(metrics_columns)}):")
        for col in sorted(metrics_columns):
            print(f"   - {col}")
        print()
    except:
        print(f"ℹ️  Table '{metrics_table}' does not exist yet (first run)\n")
        metrics_columns = []

    conn.close()

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

    print(f"  Completed {len(dataset)} predictions - Accuracy: {accuracy:.4f}")
```

**Why two tables?**
- **Predictions table**: Detailed data for segmentation ("which examples did Model A get wrong?")
- **Metrics table**: Quick comparison ("which model is best overall?")

**Rerun Safety:** If script crashes, restart it! Cached results prevent re-execution.

---

## Phase 6: Validate Results

Before analysis, check for common pitfalls:

### Run Validation Scripts:

```bash
# 1. Check for data leaks (same input_id evaluated multiple times)
python scripts/validate_benchmark.py benchmark.db predictions

# 2. Check for missing parameters
xt sql benchmark.db "SELECT COUNT(*) FROM db.predictions WHERE params_model_name IS NULL"
```

### Common Pitfalls to Check:

**Data Leakage:**
- Same `input_id` appears with different `track_id` (means it was evaluated in multiple runs)
- Solution: Use cache or check existing results before re-running

**Missing Metadata:**
- NULL values in parameter columns
- Solution: Ensure all params are in frozen dataclass

**Failed Executions:**
```sql
SELECT error, COUNT(*) FROM db.predictions WHERE error IS NOT NULL GROUP BY error
```

---

## Phase 7: Analysis with DuckDB

xetrack + DuckDB = powerful analysis:

### Teach DuckDB CLI:

```bash
# Start DuckDB UI (if duckdb >= 1.2.2)
duckdb -ui

# In the terminal or browser UI:
D INSTALL sqlite; LOAD sqlite;
D ATTACH 'benchmark.db' AS db (TYPE sqlite);
D SELECT * FROM db.predictions LIMIT 10;
```

### Common Analysis Queries:

**Aggregate by model:**
```sql
SELECT
    params_model_name,
    AVG(CASE WHEN prediction = ground_truth THEN 1.0 ELSE 0.0 END) as accuracy,
    AVG(latency) as avg_latency,
    COUNT(*) as n_predictions
FROM db.predictions
WHERE error IS NULL
GROUP BY params_model_name
ORDER BY accuracy DESC;
```

**Find errors by model:**
```sql
SELECT params_model_name, error, COUNT(*) as count
FROM db.predictions
WHERE error IS NOT NULL
GROUP BY params_model_name, error;
```

**Cache hit rate by model:**
```sql
SELECT
    params_model_name,
    COUNT(CASE WHEN cache != '' THEN 1 END) as cache_hits,
    COUNT(*) as total,
    ROUND(100.0 * COUNT(CASE WHEN cache != '' THEN 1 END) / COUNT(*), 2) as hit_rate
FROM db.predictions
GROUP BY params_model_name;
```

### Teach xetrack Python API:

**Reference:** `examples/07_data_analysis.py` for complete Reader guide

```python
from xetrack import Reader
import pandas as pd

# Read all predictions
# API Reference: Reader(db, engine='sqlite|duckdb', table='events')
reader = Reader(db='benchmark.db', engine='duckdb', table='predictions')
df = reader.to_df()  # Gets all data by default
# Or filter: df = reader.to_df(track_id='specific-run-id')

# Calculate accuracy by model
accuracy = df.groupby('params_model_name').apply(
    lambda g: (g['prediction'] == g['ground_truth']).mean()
)
print(accuracy)

# Analyze latency distribution
import matplotlib.pyplot as plt
df.boxplot(column='latency', by='params_model_name')
plt.show()
```

**Validation Checkpoint:**
- ✓ `Reader()` parameters match README documentation?
- ✓ Usage pattern matches `examples/07_data_analysis.py`?
- ✓ DataFrame columns match what `Tracker` logged?

---

### For Large Datasets: Use DuckDB Directly or Polars

**⚠️ Performance Warning:** `Reader.to_df()` uses pandas and loads entire dataset into memory. For large benchmarks (>1M rows), use:

**Option 1: Query DuckDB directly (Recommended for aggregations)**

```python
import duckdb

# Connect to database
conn = duckdb.connect()
conn.execute("INSTALL sqlite; LOAD sqlite;")
conn.execute("ATTACH 'benchmark.db' AS db (TYPE sqlite);")

# Query without loading all data
result = conn.execute("""
    SELECT
        params_model_name,
        AVG(CASE WHEN prediction = ground_truth THEN 1.0 ELSE 0.0 END) as accuracy,
        COUNT(*) as n_predictions
    FROM db.predictions
    WHERE error IS NULL
    GROUP BY params_model_name
""").fetchdf()  # Returns small aggregated DataFrame

print(result)
conn.close()
```

**Option 2: Use Polars lazy mode (Recommended for complex transformations)**

```python
import polars as pl

# Lazy query - doesn't load data until .collect()
df = pl.scan_parquet('predictions.parquet')  # or pl.read_database() if supported

# Build query lazily
result = (
    df
    .filter(pl.col('error').is_null())
    .group_by('params_model_name')
    .agg([
        ((pl.col('prediction') == pl.col('ground_truth')).mean()).alias('accuracy'),
        pl.count().alias('n_predictions')
    ])
    .collect()  # Execute only when needed
)

print(result)
```

**Export to Parquet for Polars:**

```bash
# Export from DuckDB to Parquet (more efficient than pandas)
xt sql benchmark.db "COPY (SELECT * FROM db.predictions) TO 'predictions.parquet' (FORMAT PARQUET)"
```

**When to use each:**
- **DuckDB direct**: Best for SQL-style aggregations, filtering, window functions
- **Polars lazy**: Best for complex transformations, joining multiple tables
- **Reader.to_df()**: Only for small datasets (< 100K rows) or quick prototyping

**Validation Checkpoint:**
- ✓ For large data, NOT using `Reader.to_df()`?
- ✓ Using DuckDB directly for aggregations?
- ✓ Using Polars lazy mode if complex transformations needed?

### Export Results:

```bash
# Export to CSV for sharing
xt sql benchmark.db "COPY (SELECT * FROM db.predictions) TO 'results.csv' (HEADER, DELIMITER ',')"

# Generate markdown summary
python scripts/export_summary.py benchmark.db predictions > RESULTS.md
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

## Data & Database Versioning with DVC

**Strongly recommend using DVC** for benchmarks that will be rerun or shared.

### Why DVC?

- **Data versioning**: Track dataset changes separately from code
- **Database versioning**: Version your benchmark.db results
- **Reproducibility**: Know exactly which data produced which results
- **Storage efficiency**: Git stores pointers, DVC stores actual data remotely

### Quick Setup:

```bash
# Install DVC
pip install dvc dvc-s3  # or dvc-gdrive, dvc-azure, etc.

# Initialize
dvc init

# Track your data
dvc add data/
dvc add benchmark.db

# This creates data.dvc and benchmark.db.dvc
git add data.dvc benchmark.db.dvc .dvc/.gitignore
git commit -m "feat(benchmark): add dataset and initial results"

# Push data to remote storage (not git!)
dvc remote add -d storage s3://my-bucket/xetrack-benchmarks
dvc push
```

### How Rigorous Should Versioning Be?

**Decision tree:**

#### Minimal (solo exploration, throwaway analysis)
- ✅ Git for code
- ❌ Skip DVC
- Track git hash in params (optional)

#### Standard (team experiments, production benchmarks)
- ✅ Git for code
- ✅ DVC for data + database
- ✅ Track `data.dvc` commit hash in params
- ✅ Commit after each experiment run
- Pattern: `git add benchmark.db.dvc && git commit -m "experiment: purple-mountain results"`

#### Maximum (research, audits, regulatory)
- ✅ Everything from Standard
- ✅ Branch-per-experiment pattern
- ✅ Track full git state (hash, branch, dirty status)
- ✅ Lock data.dvc during experiment (no changes mid-run)
- ✅ Push to DVC remote before merging

**Example tracker setup:**

```python
def get_data_version():
    """Get data.dvc commit hash."""
    return subprocess.check_output(
        ['git', 'log', '-n', '1', '--pretty=format:%H', '--', 'data.dvc']
    ).decode().strip()

tracker = Tracker(
    db='benchmark.db',
    engine='duckdb',
    params={
        'data_version': get_data_version(),  # Critical for reproducibility
        'git_hash': get_git_hash(),
        'experiment_name': 'purple-mountain-4392'
    }
)
```

### Database Management

**Pattern 1: Append-Only (Simplest)**
- Single `benchmark.db` file
- New experiments append rows
- Version with DVC: `dvc add benchmark.db && git commit`
- ✅ Simple, works for most cases
- ⚠️ Database grows over time

**Pattern 2: Table-per-Experiment**
- Use different table names: `predictions_exp001`, `predictions_exp002`
- Same database file
- ✅ Clean separation
- ✅ Easy to compare experiments with SQL JOINs
- ⚠️ More complex table management

```python
# Table-per-experiment
experiment_name = 'exp001'
tracker = Tracker(
    db='benchmark.db',
    table=f'predictions_{experiment_name}',
    params={'experiment': experiment_name}
)
```

**Pattern 3: Database-per-Experiment (Clean)**
- Different files: `exp001.db`, `exp002.db`
- DVC tracks each separately
- ✅ Clean separation, easy to archive
- ✅ Can delete old experiments easily
- ⚠️ Harder to compare across experiments

```python
# Database-per-experiment
tracker = Tracker(
    db=f'{experiment_name}.db',
    table='predictions'
)
```

**Recommendation:**
- Start with **Pattern 1** (append-only)
- Move to **Pattern 2** if database > 100MB or > 10 experiments
- Use **Pattern 3** for completely independent experiments

### Cleaning Up Old Experiments

```bash
# List all tracked databases
ls *.db.dvc

# Remove old experiment (DVC keeps it in cache)
dvc remove old_experiment.db.dvc
git add old_experiment.db.dvc
git commit -m "chore: archive old experiment"

# Fully delete from DVC cache (permanent!)
dvc gc --workspace --force
```

---

## Advanced: Reproducibility

For maximum reproducibility, track git state and data versions:

```python
import subprocess

def get_git_hash():
    """Get current code commit hash."""
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()

def get_data_version():
    """Get data version from DVC (if using DVC)."""
    try:
        return subprocess.check_output(
            ['git', 'log', '-n', '1', '--pretty=format:%H', '--', 'data.dvc']
        ).decode().strip()
    except:
        return None

tracker = Tracker(
    db='benchmark.db',
    engine='duckdb',
    params={
        'git_hash': get_git_hash(),
        'git_branch': subprocess.check_output(['git', 'branch', '--show-current']).decode().strip(),
        'data_version': get_data_version()  # Tracks data changes via DVC
    }
)
```

**Branch-per-Experiment Pattern** (maximum reproducibility):
1. Create experiment branch: `git checkout -b experiment/purple-mountain-4392`
2. Track branch name in params: `'git_branch': 'experiment/purple-mountain-4392'`
3. Commit code + data.dvc + benchmark.db.dvc
4. Push to remote (makes it immutable)
5. Can always `git checkout experiment/purple-mountain-4392` to reproduce exactly

**Trade-off:** High overhead, many branches. Consider if reproducibility justifies complexity.

---

## Git Tag-Based Experiment Versioning

**Recommended pattern for clean experiment history:**

### Workflow:

```python
import subprocess

def get_latest_experiment_tag():
    """Get the latest experiment tag (e*), or return e0.0.0 if none exist."""
    try:
        # Get all tags matching experiment pattern (e*)
        all_tags = subprocess.check_output(
            ['git', 'tag', '-l', 'e*']
        ).decode().strip().split('\n')

        if not all_tags or all_tags == ['']:
            return 'e0.0.0'

        # Sort and get latest
        return sorted(all_tags, key=lambda t: [int(x) for x in t.lstrip('e').split('.')])[-1]
    except:
        return 'e0.0.0'

def increment_tag(tag: str) -> str:
    """Increment patch version: e0.0.5 -> e0.0.6"""
    parts = tag.lstrip('e').split('.')
    parts[-1] = str(int(parts[-1]) + 1)
    return 'e' + '.'.join(parts)

# 1. Get next experiment version
latest_tag = get_latest_experiment_tag()  # e.g., 'e0.0.5'
next_tag = increment_tag(latest_tag)  # 'e0.0.6'

print(f"Running experiment: {next_tag}")

# 2. Run experiment with tag as parameter
tracker_predictions = Tracker(
    db='benchmark.db',
    engine='duckdb',
    table='predictions',
    params={'experiment_version': next_tag}  # Tag all rows
)

tracker_metrics = Tracker(
    db='benchmark.db',
    engine='duckdb',
    table='metrics',
    params={'experiment_version': next_tag}
)

# 3. Run your benchmark
for item in dataset:
    result = tracker_predictions.track(predict, args=[item, params])

# 4. Log aggregated metrics
tracker_metrics.log({
    'model': 'bert-base',
    'accuracy': 0.85,
    'experiment_version': next_tag  # Explicitly include tag
})

# 5. Commit database with tag
subprocess.run(['git', 'add', 'benchmark.db'], check=True)
subprocess.run(['git', 'commit', '-m', f'experiment: {next_tag} results'], check=True)

# 6. Generate tag description based on experiment
def generate_tag_description(results: dict, params: dict) -> str:
    """Generate informative tag description from experiment results."""
    parts = []

    # Model/config info
    if 'model' in params:
        parts.append(f"model={params['model']}")
    if 'learning_rate' in params:
        parts.append(f"lr={params['learning_rate']}")
    if 'batch_size' in params:
        parts.append(f"batch={params['batch_size']}")

    # Results
    if 'accuracy' in results:
        parts.append(f"acc={results['accuracy']:.4f}")
    if 'loss' in results:
        parts.append(f"loss={results['loss']:.4f}")

    # Data info (if tracked)
    data_version = subprocess.check_output(
        ['git', 'log', '-n', '1', '--pretty=format:%h', '--', 'data.dvc']
    ).decode().strip()
    if data_version:
        parts.append(f"data={data_version[:7]}")

    return ' | '.join(parts)

# Auto-generate description
tag_description = generate_tag_description(
    results={'accuracy': 0.85},
    params={'model': 'bert-base', 'learning_rate': 1e-4}
)
# Example: "model=bert-base | lr=0.0001 | acc=0.8500 | data=a3f2b1c"

# Let user review/override
print(f"\n📝 Suggested tag description:")
print(f"   {tag_description}")
response = input("   Use this description? [Y/n/edit]: ").strip().lower()

if response == 'edit':
    tag_description = input("   Enter custom description: ").strip()
elif response == 'n':
    tag_description = f"Experiment {next_tag}"

# Create annotated tag with description
subprocess.run([
    'git', 'tag', '-a', next_tag,
    '-m', tag_description
], check=True)

print(f"✅ Experiment {next_tag} complete and tagged!")
print(f"   Description: {tag_description}")
```

### Complete Workflow: DVC + Git + Commit Hash Tracking

Here's the full end-to-end workflow for tracking experiments with DVC versioning:

```python
#!/usr/bin/env python3
"""Complete experiment workflow with DVC tracking and commit hash recording."""
import subprocess

# 1. Run your experiment (as shown above)
# ... experiment code ...

# 2. Track database with DVC
subprocess.run(['dvc', 'add', 'benchmark.db'], check=True)
print("✅ Database tracked with DVC")

# 3. Add .dvc file to git (NOT the database itself!)
subprocess.run(['git', 'add', 'benchmark.db.dvc', '.dvc/.gitignore'], check=True)

# 4. Commit the .dvc file
commit_msg = f"experiment: {next_tag} results"
subprocess.run(['git', 'commit', '-m', commit_msg], check=True)

# 5. Get the commit hash of this commit
db_commit_hash = subprocess.check_output(
    ['git', 'rev-parse', 'HEAD']
).decode().strip()

# 6. Get data version (commit hash when data.dvc was last modified)
try:
    data_commit_hash = subprocess.check_output(
        ['git', 'log', '-n', '1', '--pretty=format:%H', '--', 'data.dvc']
    ).decode().strip()
except:
    data_commit_hash = None

# 7. Record commit hashes in a tracking file or database
# Option A: Add to metrics table
tracker_metrics.log({
    'experiment_version': next_tag,
    'db_commit': db_commit_hash[:7],  # Short hash
    'data_commit': data_commit_hash[:7] if data_commit_hash else None,
    'accuracy': 0.85,
    'model': 'bert-base'
})

# Option B: Save to tracking file
with open('experiment_log.txt', 'a') as f:
    f.write(f"{next_tag}|{db_commit_hash}|{data_commit_hash}|{tag_description}\n")

# 8. Create git tag with auto-generated description
subprocess.run([
    'git', 'tag', '-a', next_tag,
    '-m', tag_description
], check=True)

# 9. Push database to DVC remote storage
subprocess.run(['dvc', 'push'], check=True)

# 10. Push git commits and tags
subprocess.run(['git', 'push', 'origin', 'main'], check=True)
subprocess.run(['git', 'push', 'origin', next_tag], check=True)

print(f"""
✅ Experiment {next_tag} complete!

   Database commit: {db_commit_hash[:7]}
   Data version:    {data_commit_hash[:7] if data_commit_hash else 'N/A'}
   Description:     {tag_description}

   To reproduce later:
     git checkout {next_tag}
     dvc pull
""")
```

### Checking File Version Hashes

Useful commands for tracking version information:

```bash
# Get commit hash of current HEAD
git rev-parse HEAD

# Get short hash (7 characters)
git rev-parse --short HEAD

# Get commit hash when specific file was last changed
git log -n 1 --pretty=format:%H -- benchmark.db.dvc
git log -n 1 --pretty=format:%H -- data.dvc

# Get commit hash at specific tag
git rev-parse e0.0.3

# Check if file has changed since last commit
git diff --quiet benchmark.db.dvc && echo "No changes" || echo "Modified"

# Get DVC file hash (MD5 of actual data)
cat benchmark.db.dvc | grep md5
```

### Why Track Both Hashes?

1. **Code commit hash** (`git rev-parse HEAD`):
   - **Captures entire repository state** - code, .dvc files, everything
   - Guarantees exact reproducibility of the experiment
   - Used for `git checkout <hash>` to restore complete state

2. **Data commit hash** (`git log -n 1 ... -- data.dvc`):
   - **Captures only when data.dvc changed** - tracks data versions independently
   - Multiple experiments can share same data version (different code, same data)
   - Useful for identifying "same data, different model" comparisons

**Critical purpose:** By tracking data.dvc commit separately, you can detect if data changed between experiments without comparing entire repo state.

### Recommended: Schema Validation Before Experiments

**Critical check:** Detect parameter renames or schema drift before running experiments.

**Problem:** If you rename a parameter in code, xetrack creates a NEW column instead of reusing the old one:

```python
# Experiment 1
params = {'learning_rate': 0.001}  # Creates column 'learning_rate'

# Experiment 2 - renamed parameter (BUG!)
params = {'lr': 0.001}  # Creates NEW column 'lr'
# Old data in 'learning_rate', new data in 'lr' - split across columns!
```

**Solution:** Validate schema before running experiments:

```python
from xetrack import Reader
import sqlite3
from dataclasses import fields
from difflib import get_close_matches

def validate_schema_before_experiment(db_path, table, new_params_dataclass):
    """
    Compare current database schema with new experiment parameters.
    Detect potential issues: renamed params, similar names, missing columns.
    """
    # 1. Get current schema from database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    existing_columns = {row[1] for row in cursor.fetchall()}
    conn.close()

    # 2. Extract parameter names from new dataclass
    if hasattr(new_params_dataclass, '__dataclass_fields__'):
        # It's a dataclass
        new_param_names = {f'params_{f.name}' for f in fields(new_params_dataclass)}
    else:
        # It's a dict or object
        new_param_names = {f'params_{k}' for k in new_params_dataclass.__dict__.keys()}

    # 3. Detect potential issues
    issues = []

    # Check for renamed parameters (similar names)
    for new_param in new_param_names:
        if new_param not in existing_columns:
            # Find similar column names (potential renames)
            similar = get_close_matches(new_param, existing_columns, n=1, cutoff=0.6)
            if similar:
                issues.append({
                    'type': 'POTENTIAL_RENAME',
                    'new_param': new_param,
                    'old_param': similar[0],
                    'similarity': 'high'
                })

    # Check for missing parameters (were in schema, not in new code)
    param_columns = {col for col in existing_columns if col.startswith('params_')}
    missing_params = param_columns - new_param_names
    if missing_params:
        issues.append({
            'type': 'MISSING_PARAMS',
            'params': missing_params
        })

    # 4. Report issues and get user input
    if issues:
        print("\n⚠️  SCHEMA VALIDATION ISSUES DETECTED:\n")

        for issue in issues:
            if issue['type'] == 'POTENTIAL_RENAME':
                print(f"❌ Potential parameter rename detected:")
                print(f"   Old column: {issue['old_param']}")
                print(f"   New param:  {issue['new_param']}")
                print(f"\n   This will create a NEW column, splitting data across two columns!")
                print(f"\n   Options:")
                print(f"   1. Rename column in database: ALTER TABLE {table} RENAME COLUMN {issue['old_param']} TO {issue['new_param']}")
                print(f"   2. Change code to use old name: {issue['old_param']}")
                print(f"   3. Confirm this is intentional (creates new column)\n")

            elif issue['type'] == 'MISSING_PARAMS':
                print(f"⚠️  Parameters from previous experiments missing in new code:")
                for param in issue['params']:
                    print(f"   - {param}")
                print(f"\n   If these were renamed, use ALTER TABLE to rename columns.")
                print(f"   If intentionally removed, this is OK (old data preserved).\n")

        print("   Actions:")
        print("   [r] Rename column in database (recommended if parameter was renamed)")
        print("   [c] Continue anyway (will create new columns)")
        print("   [a] Abort (fix code first)")

        choice = input("\n   Your choice: ").strip().lower()

        if choice == 'r':
            # Help user rename column
            for issue in issues:
                if issue['type'] == 'POTENTIAL_RENAME':
                    old_col = issue['old_param']
                    new_col = issue['new_param']
                    print(f"\n📝 Renaming {old_col} → {new_col}")

                    # Execute rename in SQLite
                    conn = sqlite3.connect(db_path)
                    conn.execute(f"ALTER TABLE {table} RENAME COLUMN {old_col} TO {new_col}")
                    conn.commit()
                    conn.close()

                    print(f"✅ Column renamed successfully!")

            print("\n✅ Schema updated. Safe to run experiment.")
            return True

        elif choice == 'c':
            print("⚠️  Continuing with schema drift. New columns will be created.")
            return True

        else:  # 'a' or anything else
            print("❌ Experiment aborted. Fix code or schema first.")
            return False

    else:
        print("✅ Schema validation passed - no issues detected")
        return True

# Usage before running experiment:
if not validate_schema_before_experiment('benchmark.db', 'predictions', ModelParams):
    exit(1)  # Don't run experiment if validation failed
```

**Example output:**

```
⚠️  SCHEMA VALIDATION ISSUES DETECTED:

❌ Potential parameter rename detected:
   Old column: params_learning_rate
   New param:  params_lr

   This will create a NEW column, splitting data across two columns!

   Options:
   1. Rename column in database: ALTER TABLE predictions RENAME COLUMN params_learning_rate TO params_lr
   2. Change code to use old name: params_learning_rate
   3. Confirm this is intentional (creates new column)

   Actions:
   [r] Rename column in database (recommended if parameter was renamed)
   [c] Continue anyway (will create new columns)
   [a] Abort (fix code first)

   Your choice: r

📝 Renaming params_learning_rate → params_lr
✅ Column renamed successfully!
✅ Schema updated. Safe to run experiment.
```

**Why this matters:**
- **Prevents data fragmentation** - Keeps related data in one column
- **Maintains clean schema** - Avoids accumulating renamed columns
- **SQLite flexibility** - Easy to rename columns with ALTER TABLE
- **Catches mistakes early** - Before running expensive experiments

**Best practice workflow:**
1. Make code changes
2. Run schema validation (detects renames)
3. Fix schema or code
4. Run experiment with clean schema

### Recommended: Pre-Run Data Check

**Best practice:** Verify data is committed before running experiments to ensure reproducibility:

```python
def check_data_committed():
    """
    Optional safety check: warn if data.dvc has uncommitted changes.
    Helps prevent running experiments with unversioned data.
    """
    try:
        # Check if data.dvc has uncommitted changes
        result = subprocess.run(
            ['git', 'diff', '--quiet', 'data.dvc'],
            capture_output=True
        )
        if result.returncode != 0:
            print(
                "⚠️  WARNING: data.dvc has uncommitted changes!\n"
                "   Recommended: commit data changes before running experiments:\n"
                "   1. dvc add data/\n"
                "   2. git add data.dvc\n"
                "   3. git commit -m 'data: updated dataset'\n"
                "\n"
                "   This ensures reproducibility - every experiment will have\n"
                "   a committed data version.\n"
            )
            response = input("   Continue anyway? [y/N]: ").strip().lower()
            if response != 'y':
                raise RuntimeError("Experiment cancelled by user")

        # Also check if data.dvc is staged but not committed
        result = subprocess.run(
            ['git', 'diff', '--cached', '--quiet', 'data.dvc'],
            capture_output=True
        )
        if result.returncode != 0:
            print(
                "⚠️  WARNING: data.dvc is staged but not committed!\n"
                "   Recommended: git commit -m 'data: updated dataset'\n"
            )

        print("✅ Data is committed - good for reproducibility")
        return True
    except FileNotFoundError:
        print("ℹ️  data.dvc not found (may not be using DVC for data)")
        return True

# Optional: call this before running experiments
# check_data_committed()
```

**Why this helps:**
- Prevents "ghost experiments" with unversioned data
- Improves reproducibility - every result can be traced to exact data version
- Catches mistakes where data changed but wasn't committed

**Note:** This check is optional. For quick prototyping or debugging, you may want to skip it.

**Example use case:**
```python
# In your benchmark params
def get_versions():
    """Get both commit hashes for tracking."""
    code_commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD']
    ).decode().strip()

    data_commit = subprocess.check_output(
        ['git', 'log', '-n', '1', '--pretty=format:%H', '--', 'data.dvc']
    ).decode().strip()

    return code_commit[:7], data_commit[:7]

# Before running experiment
check_data_committed()

code_commit, data_commit = get_versions()
params = {
    'model': 'bert-base',
    'learning_rate': 0.0001,
    'code_commit': code_commit,      # Entire repo state
    'data_commit': data_commit       # Just data version
}
```

This lets you later query:
```sql
-- Find all experiments using the same data version
-- (useful to compare "same data, different hyperparameters")
SELECT experiment_version, model, accuracy, code_commit, data_commit
FROM metrics
WHERE data_commit = '3a2f1b'  -- Same data across all these experiments
ORDER BY accuracy DESC;

-- Find experiments where ONLY data changed (code stayed same)
SELECT experiment_version, accuracy, data_commit
FROM metrics
WHERE code_commit = 'a1b2c3d'  -- Same code
ORDER BY data_commit;
```

### Benefits:

1. **Clear separation**: `e*` tags for experiments, `v*` tags for code releases
2. **No conflicts**: Won't interfere with semantic versioning (v1.0.0, v1.1.0, etc.)
3. **Traceable results**: Each tag corresponds to exact code + data + results
4. **Easy comparison**: Query by tag to compare experiments
5. **Reproducible**: Checkout tag to get exact state

### View experiment history with descriptions:

```bash
# List only experiment tags (not version tags)
git tag -l 'e*' -n9

# Example output:
# e0.0.1          model=logistic | lr=0.001 | acc=0.8200 | data=3a2f1b
# e0.0.2          model=bert-base | lr=0.0001 | acc=0.8500 | data=3a2f1b
# e0.0.3          model=bert-base | lr=0.0001 | acc=0.8900 | data=7c4e2a (new dataset)
# e0.0.4          model=roberta | lr=0.00005 | acc=0.9100 | data=7c4e2a

# Meanwhile, your code versions remain separate:
git tag -l 'v*'
# v1.0.0          Initial release
# v1.1.0          Add new feature

# Show specific experiment details
git show e0.0.3
```

### Query by experiment version:

```sql
-- Compare experiments
SELECT
    experiment_version,
    AVG(CASE WHEN prediction = ground_truth THEN 1.0 ELSE 0.0 END) as accuracy
FROM db.predictions
WHERE experiment_version IN ('e0.0.1', 'e0.0.2')
GROUP BY experiment_version;

-- Get metrics for specific experiment
SELECT * FROM db.metrics WHERE experiment_version = 'e0.0.3';
```

### With DVC:

```bash
# After tagging, push database to DVC
dvc add benchmark.db
git add benchmark.db.dvc
git commit --amend --no-edit  # Add to same commit
dvc push

# Later, reproduce experiment
git checkout e0.0.3
dvc pull  # Gets exact database state
```

### Automation Script:

Create `run_experiment.py`:

```python
#!/usr/bin/env python
"""
Run a new experiment with automatic versioning.

Usage:
    python run_experiment.py --model bert-base --data dataset.csv
"""

import subprocess
import argparse
from your_benchmark import run_benchmark

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    args = parser.parse_args()

    # Get next version
    latest = get_latest_tag()
    next_tag = increment_tag(latest)

    print(f"🚀 Running experiment {next_tag}")

    # Run benchmark
    results = run_benchmark(
        model=args.model,
        data=args.data,
        experiment_version=next_tag
    )

    # Generate informative tag description
    tag_desc = f"model={args.model} | data={args.data} | acc={results['accuracy']:.4f}"

    # Commit and tag
    subprocess.run(['git', 'add', 'benchmark.db'], check=True)
    subprocess.run(['git', 'commit', '-m', f'experiment: {next_tag} - {args.model}'], check=True)
    subprocess.run(['git', 'tag', '-a', next_tag, '-m', tag_desc], check=True)

    print(f"✅ Experiment complete!")
    print(f"   Tag: {next_tag}")
    print(f"   Accuracy: {results['accuracy']:.4f}")

if __name__ == "__main__":
    main()
```

**Trade-off:** Requires discipline to always use the workflow. Consider automation or git hooks.

---

## References

For deeper guidance, see:

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
