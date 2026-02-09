# Experiment Design

Phase 0 (Ideation) and Phase 1 (Design) of the benchmark workflow. Referenced from SKILL.md.

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
