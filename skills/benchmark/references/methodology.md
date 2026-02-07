# Benchmarking Methodology

Core principles for rigorous ML/AI benchmarking.

## The Single-Execution Principle

**Every datapoint should execute exactly once.** Not once per metric. Not once per analysis pass. Once.

This is the foundation of reproducible benchmarks. When a datapoint can be executed multiple times:
- You waste compute and time
- Results become inconsistent
- Cache effectiveness drops
- Analysis becomes unreliable

## Design End-to-Start

Before writing any code, understand:

### 1. Questions You Want to Answer
- "Is Model A better than Model B?"
- "Which hyperparameters work best?"
- "Does preprocessing improve accuracy?"

These questions determine what data you need to collect.

### 2. Segmentations You'll Perform
- By model type?
- By data subset?
- By prompt variation?
- By difficulty level?

If you don't log the metadata for these cuts, no amount of clever analysis will save you later.

### 3. Metrics That Matter
- Accuracy, F1, precision, recall?
- Latency, throughput?
- Cost per prediction?
- Token usage?

**Critical:** Always save raw outputs (full probabilities, complete LLM responses), not just processed results. You can always re-process. You can't reconstruct what the model actually said.

## Stateless Function Design

Your single-execution function must be:

**1. Stateless**
No shared mutable state. Bad:
```python
results = []  # Global - race condition!

def predict(x):
    result = model(x)
    results.append(result)  # Don't do this
    return result
```

Good:
```python
def predict(x):
    return model(x)  # Pure function
```

**2. Deterministic**
Same inputs → same outputs (modulo randomness you control with seeds)

**3. Thread/Process Safe**
Can be called in parallel without corruption

**4. Comprehensive**
Returns everything you might need, including:
- The prediction
- Raw model outputs
- Metadata (latency, tokens, cost)
- Error information (if failed)

## Frozen Dataclasses for Parameters

Use `@dataclass(frozen=True, slots=True)` for all parameters:

```python
@dataclass(frozen=True, slots=True)
class BenchmarkParams:
    model_name: str
    temperature: float
    max_tokens: int
    system_prompt: str
```

**Why frozen?**
- Makes params hashable → enables caching
- Immutable → prevents accidental modification
- Forces explicit parameter changes

**Why slots?**
- 30-40% memory reduction for large param grids
- Faster attribute access
- More compact representation

**Rules:**
- Use immutable types only (no lists/dicts)
- Use tuples instead of lists
- Use frozensets instead of sets
- Round floats if using as cache keys

## Caching as Correctness

Caching is not an optimization—it's a correctness tool.

**Without caching:**
- Script crashes → re-run everything
- Add new parameter → re-run everything
- Change analysis → re-run everything

**With caching:**
- Script crashes → resume where you left off
- Add new parameter → only run new combinations
- Change analysis → instant, no re-execution

**Cache Requirements:**
- All arguments must be hashable
- Use frozen dataclasses
- Avoid floats (or quantize them)
- No lists/dicts in arguments

## Failure as First-Class Data

Failures are not edge cases—they're signals.

**Bad:**
```python
try:
    prediction = model(input)
except:
    continue  # Silent failure
```

**Good:**
```python
try:
    prediction = model(input)
    error = None
except Exception as e:
    prediction = None
    error = str(e)

return {'prediction': prediction, 'error': error}
```

Track failures with reason codes:
- Rate limit errors
- Timeout errors
- Malformed input errors
- Model errors

A failed call with a reason is infinitely more useful than a missing row.

## Two-Layer Storage

Store data at two levels:

**Layer 1: Prediction-Level**
Every single prediction with full metadata:
- `input_id`, `prediction`, `ground_truth`
- `latency`, `tokens`, `cost`
- `error` (if failed)
- All parameter values (unpacked from dataclass)

**Layer 2: Experiment-Level** (Optional)
Aggregated metrics per parameter combination:
- `model_name`, `accuracy`, `avg_latency`
- `total_cost`, `error_rate`
- Summary statistics

Layer 1 enables segmentation analysis later. Layer 2 enables quick comparisons.

## Validation Checklist

Before trusting results:

✅ **No data leaks** - Same `input_id` only appears once per experiment
✅ **No duplicates** - No duplicate `(track_id, input_id)` pairs
✅ **No missing params** - All parameter columns are populated
✅ **Failures captured** - Error column exists and is populated for failures
✅ **Cache working** - High cache hit rate on re-runs

Run: `python scripts/validate_benchmark.py <db> <table>`

## When NOT to Follow This

Not every benchmark needs industrial-grade infrastructure:

**Skip the rigor when:**
- Quick one-off comparison (< 5 minutes to re-run)
- Prototyping phase (speed > reproducibility)
- Solo exploration (no team coordination needed)
- Throwaway analysis

**Use the rigor when:**
- Results will be shared or published
- Experiment takes > 10 minutes to run
- Testing expensive APIs (LLMs, cloud services)
- Team collaboration (multiple people running experiments)
- Reproducibility matters (research, production)

## Memory Aids

**Remember:**
- Storage is cheap; regret is expensive
- Design backwards from questions you want to answer
- Single-execution is a correctness constraint, not optimization
- Cache everything; analyze freely
- Failures are data, not noise

**The Goal:**
Implement once, run once, analyze without pain.
