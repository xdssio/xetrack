# DuckDB Analysis Recipes

Powerful SQL queries for analyzing xetrack benchmark results with DuckDB.

## Setup

```bash
# Install DuckDB CLI
curl https://install.duckdb.org | sh

# Start DuckDB UI (if version >= 1.2.2)
duckdb -ui

# Or use terminal interface
duckdb
```

## Connecting to xetrack Database

```sql
-- Load SQLite extension
INSTALL sqlite;
LOAD sqlite;

-- Attach your benchmark database
ATTACH 'benchmark.db' AS db (TYPE sqlite);

-- List tables
SHOW TABLES FROM db;

-- Inspect schema
DESCRIBE db.predictions;
```

## Common Queries

### 1. Overall Performance by Model

```sql
SELECT
    params_model_name,
    COUNT(*) as n_predictions,
    AVG(CASE WHEN prediction = ground_truth THEN 1.0 ELSE 0.0 END) as accuracy,
    AVG(latency) as avg_latency_sec,
    MIN(latency) as min_latency,
    MAX(latency) as max_latency,
    STDDEV(latency) as std_latency
FROM db.predictions
WHERE error IS NULL  -- Exclude failed predictions
GROUP BY params_model_name
ORDER BY accuracy DESC;
```

### 2. Error Analysis

```sql
-- Error counts by type
SELECT
    params_model_name,
    error,
    COUNT(*) as error_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY params_model_name), 2) as pct_of_model_errors
FROM db.predictions
WHERE error IS NOT NULL AND error != ''
GROUP BY params_model_name, error
ORDER BY params_model_name, error_count DESC;
```

### 3. Latency Percentiles

```sql
SELECT
    params_model_name,
    APPROX_QUANTILE(latency, 0.50) as p50_latency,
    APPROX_QUANTILE(latency, 0.90) as p90_latency,
    APPROX_QUANTILE(latency, 0.95) as p95_latency,
    APPROX_QUANTILE(latency, 0.99) as p99_latency
FROM db.predictions
WHERE error IS NULL
GROUP BY params_model_name;
```

### 4. Cache Effectiveness

```sql
SELECT
    params_model_name,
    COUNT(*) as total,
    SUM(CASE WHEN cache != '' THEN 1 ELSE 0 END) as cache_hits,
    SUM(CASE WHEN cache = '' THEN 1 ELSE 0 END) as cache_misses,
    ROUND(100.0 * SUM(CASE WHEN cache != '' THEN 1 ELSE 0 END) / COUNT(*), 2) as hit_rate_pct
FROM db.predictions
GROUP BY params_model_name;
```

### 5. Performance Over Time

```sql
-- Check if performance degrades over time
SELECT
    params_model_name,
    DATE_TRUNC('hour', CAST(timestamp AS TIMESTAMP)) as hour,
    AVG(CASE WHEN prediction = ground_truth THEN 1.0 ELSE 0.0 END) as accuracy,
    AVG(latency) as avg_latency,
    COUNT(*) as n_predictions
FROM db.predictions
WHERE error IS NULL
GROUP BY params_model_name, hour
ORDER BY params_model_name, hour;
```

### 6. Compare Two Models Head-to-Head

```sql
WITH model_a AS (
    SELECT input_id, prediction, latency
    FROM db.predictions
    WHERE params_model_name = 'bert-base' AND error IS NULL
),
model_b AS (
    SELECT input_id, prediction, latency
    FROM db.predictions
    WHERE params_model_name = 'distilbert' AND error IS NULL
)
SELECT
    COUNT(*) as shared_inputs,
    SUM(CASE WHEN a.prediction = b.prediction THEN 1 ELSE 0 END) as agreement,
    ROUND(100.0 * SUM(CASE WHEN a.prediction = b.prediction THEN 1 ELSE 0 END) / COUNT(*), 2) as agreement_pct,
    AVG(a.latency) as model_a_avg_latency,
    AVG(b.latency) as model_b_avg_latency
FROM model_a a
JOIN model_b b ON a.input_id = b.input_id;
```

### 7. Find Hardest Examples

```sql
-- Find inputs where all models failed or got wrong
WITH model_performance AS (
    SELECT
        input_id,
        COUNT(*) as n_models_tested,
        SUM(CASE WHEN prediction = ground_truth THEN 1 ELSE 0 END) as n_correct,
        SUM(CASE WHEN error IS NOT NULL AND error != '' THEN 1 ELSE 0 END) as n_errors,
        ANY_VALUE(ground_truth) as ground_truth
    FROM db.predictions
    GROUP BY input_id
)
SELECT *
FROM model_performance
WHERE n_correct = 0  -- No model got it right
ORDER BY n_models_tested DESC
LIMIT 20;
```

### 8. Hyperparameter Search Results

```sql
-- Best hyperparameters by accuracy
SELECT
    params_model_name,
    params_temperature,
    params_max_tokens,
    AVG(CASE WHEN prediction = ground_truth THEN 1.0 ELSE 0.0 END) as accuracy,
    AVG(latency) as avg_latency,
    COUNT(*) as n_predictions
FROM db.predictions
WHERE error IS NULL
GROUP BY params_model_name, params_temperature, params_max_tokens
ORDER BY accuracy DESC, avg_latency ASC
LIMIT 10;
```

### 9. Cost Analysis (if tracked)

```sql
-- Assuming you track 'cost_usd' column
SELECT
    params_model_name,
    SUM(cost_usd) as total_cost_usd,
    AVG(cost_usd) as avg_cost_per_prediction,
    COUNT(*) as n_predictions,
    SUM(cost_usd) / SUM(CASE WHEN prediction = ground_truth THEN 1 ELSE 0 END) as cost_per_correct_prediction
FROM db.predictions
WHERE error IS NULL
GROUP BY params_model_name
ORDER BY total_cost_usd DESC;
```

### 10. Data Leak Detection

```sql
-- Find input_ids evaluated multiple times with different track_ids
SELECT
    input_id,
    COUNT(DISTINCT track_id) as n_different_runs,
    COUNT(*) as total_evaluations,
    STRING_AGG(DISTINCT params_model_name, ', ') as models_used
FROM db.predictions
GROUP BY input_id
HAVING COUNT(DISTINCT track_id) > 1
ORDER BY n_different_runs DESC;
```

## Window Functions

### Rolling Average Accuracy

```sql
SELECT
    params_model_name,
    timestamp,
    prediction,
    ground_truth,
    AVG(CASE WHEN prediction = ground_truth THEN 1.0 ELSE 0.0 END)
        OVER (PARTITION BY params_model_name ORDER BY timestamp ROWS BETWEEN 99 PRECEDING AND CURRENT ROW)
        as rolling_accuracy_last_100
FROM db.predictions
WHERE error IS NULL
ORDER BY params_model_name, timestamp;
```

### Rank Models by Performance

```sql
SELECT
    params_model_name,
    AVG(CASE WHEN prediction = ground_truth THEN 1.0 ELSE 0.0 END) as accuracy,
    RANK() OVER (ORDER BY AVG(CASE WHEN prediction = ground_truth THEN 1.0 ELSE 0.0 END) DESC) as rank_by_accuracy,
    AVG(latency) as avg_latency,
    RANK() OVER (ORDER BY AVG(latency) ASC) as rank_by_speed
FROM db.predictions
WHERE error IS NULL
GROUP BY params_model_name;
```

## Exporting Results

### Export to CSV

```sql
-- Export summary to CSV
COPY (
    SELECT
        params_model_name,
        AVG(CASE WHEN prediction = ground_truth THEN 1.0 ELSE 0.0 END) as accuracy,
        AVG(latency) as avg_latency,
        COUNT(*) as n_predictions
    FROM db.predictions
    WHERE error IS NULL
    GROUP BY params_model_name
) TO 'results.csv' (HEADER, DELIMITER ',');
```

### Export to Parquet

```sql
-- More efficient for large datasets
COPY (SELECT * FROM db.predictions) TO 'predictions.parquet' (FORMAT PARQUET);
```

## Tips

1. **Use EXPLAIN for slow queries:**
   ```sql
   EXPLAIN SELECT ... FROM db.predictions WHERE ...;
   ```

2. **Create views for common queries:**
   ```sql
   CREATE VIEW model_summary AS
   SELECT
       params_model_name,
       AVG(CASE WHEN prediction = ground_truth THEN 1.0 ELSE 0.0 END) as accuracy
   FROM db.predictions
   WHERE error IS NULL
   GROUP BY params_model_name;

   SELECT * FROM model_summary;
   ```

3. **Use CTEs for readable complex queries:**
   ```sql
   WITH successful_predictions AS (
       SELECT * FROM db.predictions WHERE error IS NULL
   ),
   aggregated AS (
       SELECT params_model_name, AVG(latency) as avg_latency
       FROM successful_predictions
       GROUP BY params_model_name
   )
   SELECT * FROM aggregated ORDER BY avg_latency;
   ```

4. **Filter early for performance:**
   ```sql
   -- Good: filter before aggregation
   SELECT params_model_name, COUNT(*)
   FROM db.predictions
   WHERE error IS NULL  -- Filter early
   GROUP BY params_model_name;
   ```

5. **Use APPROX functions for large datasets:**
   ```sql
   -- Faster approximations for huge tables
   SELECT
       APPROX_COUNT_DISTINCT(input_id) as unique_inputs,
       APPROX_QUANTILE(latency, 0.95) as p95_latency
   FROM db.predictions;
   ```
