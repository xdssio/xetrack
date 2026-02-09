# Development Phase vs Experiment Phase

Detailed guidance on the two distinct phases of benchmarking. Referenced from SKILL.md workflow overview.

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
- ✅ **Code structure validated** (no chaotic functions that affect analysis)
- ✅ Code committed to git

### Code Structure for Benchmarking

**Keep prediction functions simple - complexity leads to unusable results.**

If your function is chaotic, refactor before experiments. Complex benchmarking code causes:
- ❌ Hard to debug failures (can't isolate what went wrong)
- ❌ Hard to compare results (logic mixed with I/O)
- ❌ Hard to analyze errors (which component failed?)
- ❌ Hard to trust results (too many moving parts)

**Recommended structure (for benchmarking):**

```python
# ✅ GOOD: Simple, clear separation

def predict(item: dict, params: ModelParams) -> dict:
    """
    Keep this function SIMPLE and FOCUSED on prediction logic.
    Complexity = bugs = unusable benchmark results.
    """
    try:
        # Step 1: Extract what you need
        text = item['text']

        # Step 2: Call your model/function
        response = model.predict(text, temperature=params.temperature)

        # Step 3: Return EVERYTHING (don't parse/filter)
        return {
            'input_id': item['id'],
            'raw_response': response,        # ← CRITICAL: Save raw output
            'prediction': extract_answer(response),
            'ground_truth': item['label'],
            'error': None
        }

    except Exception as e:
        # Step 4: Handle errors gracefully
        return {
            'input_id': item.get('id'),
            'raw_response': None,
            'prediction': None,
            'ground_truth': item.get('label'),
            'error': str(e),
            'error_type': type(e).__name__
        }

# If your function is >50 lines, extract helpers:
def prepare_text(item):
    return item['text'].strip()

def extract_answer(response):
    # Parsing logic here
    return parsed_answer
```

**Anti-patterns that break benchmarks:**

```python
# ❌ BAD: Mixing I/O with logic
def predict(item_id, params):  # Takes ID, not item
    item = load_from_database(item_id)  # I/O inside function
    response = model.predict(item['text'])
    save_to_file(response)  # Side effect!
    return parse(response)

# Problems:
# - Can't test without database
# - Can't cache (item_id not hashable dataclass)
# - Side effects make debugging hard
# - What if save_to_file fails? Data lost!

# ❌ BAD: Complex preprocessing inside
def predict(item, params):
    # 100 lines of preprocessing
    text = item['text']
    text = clean(text)
    text = normalize(text)
    text = augment(text)
    text = tokenize(text)
    # ... 95 more lines ...
    return model.predict(text)

# Problems:
# - Can't isolate preprocessing bugs
# - Hard to test model separately
# - Can't reprocess differently later
# - Which step caused the error?

# ❌ BAD: Filtering/parsing too early
def predict(item, params):
    response = model.predict(item['text'])

    # Only save if confident
    if response['confidence'] > 0.8:  # ← Loses data!
        return {'prediction': response['answer']}  # ← Only parsed output!
    else:
        return None  # ← Missing low-confidence examples!

# Problems:
# - Lost raw response (can't reprocess)
# - Lost low-confidence data (can't analyze)
# - Hard threshold decision (can't change later)
```

**Benchmarking-specific recommendations:**

1. **Load data OUTSIDE the prediction function**
   ```python
   # ✅ GOOD
   items = load_all_data()  # Once, outside loop
   for item in items:
       result = tracker.track(predict, args=[item, params])
   ```

2. **Keep prediction function PURE (input → output, no side effects)**
   ```python
   # ✅ GOOD: Pure function
   def predict(item, params):
       return {'prediction': ..., 'raw_response': ...}

   # ❌ BAD: Side effects
   def predict(item, params):
       save_to_log(item)  # Side effect
       update_global_counter()  # Side effect
       return result
   ```

3. **Save RAW outputs, parse during analysis**
   ```python
   # ✅ GOOD: Save everything
   return {'raw_response': response, 'prediction': parsed}

   # ❌ BAD: Only save parsed
   return {'prediction': parsed}  # Can't reprocess!
   ```

4. **If function is complex, extract to helpers**
   ```python
   # If predict() is >50 lines, break it down:
   def prepare_input(item): ...
   def call_model(text, params): ...
   def parse_output(response): ...

   def predict(item, params):
       # Just orchestrate
       text = prepare_input(item)
       response = call_model(text, params)
       parsed = parse_output(response)
       return {'raw_response': response, 'prediction': parsed}
   ```

**Why this matters for benchmarking:**
- Simple functions = easy to debug = trustworthy results
- Pure functions = reproducible = cacheable
- Raw data saved = reprocessable = flexible analysis
- Clear errors = understandable failures = actionable insights

### 🔬 Experiment Phase (Production Runs)

**Purpose:** Run full, reproducible experiments for analysis.

**Rules:**
- ⛔ **NO CODE CHANGES during experiments** - If you need to change code, go back to dev phase
- ✅ Run on full dataset
- ✅ Track everything (code commit, data commit, timestamps)
- ✅ Use production database/table (e.g., `predictions`, `metrics`)
- ✅ Data must be committed (DVC)
- ✅ Create git tags for each experiment

**Helper: Check data is committed (DVC users)**

```python
import subprocess

def check_data_committed():
    """Ensure data.dvc is committed - prevents running experiments with uncommitted data."""
    try:
        result = subprocess.run(
            ['git', 'diff', '--quiet', 'data.dvc'],
            capture_output=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                "❌ ERROR: data.dvc has uncommitted changes!\n"
                "   You MUST commit data changes before running experiments:\n"
                "   1. dvc add data/\n"
                "   2. git add data.dvc\n"
                "   3. git commit -m 'data: updated dataset'\n"
                "\n"
                "   This ensures reproducibility - every experiment must have\n"
                "   a committed data version."
            )
        result = subprocess.run(
            ['git', 'diff', '--cached', '--quiet', 'data.dvc'],
            capture_output=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                "❌ ERROR: data.dvc is staged but not committed!\n"
                "   Run: git commit -m 'data: updated dataset'"
            )
        print("✅ Data is committed - safe to run experiment")
        return True
    except FileNotFoundError:
        print("⚠️  data.dvc not found (may not be using DVC for data)")
        return True
```

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
