# Xetrack Examples

Comprehensive examples demonstrating all features of xetrack.

## Quick Start

Run all examples:
```bash
python examples/run_all.py
```

Run individual examples:
```bash
python examples/01_quickstart.py
python examples/02_track_functions.py
# ... etc
```

## Examples Overview

### 1. Quickstart (`01_quickstart.py`)
Basic usage of xetrack:
- Creating a tracker
- Setting params
- Logging metrics
- Converting to DataFrame
- Multiple experiment types

**Concepts:** tracker, log(), params, to_df(), tables

---

### 2. Track Functions (`02_track_functions.py`)
Function execution tracking:
- Using `tracker.track()` to monitor functions
- Using `@tracker.wrap()` decorator
- System and network monitoring
- Tracking function time, args, kwargs

**Concepts:** track(), wrap(), system_params, network_params

---

### 3. Dataclass Auto-Unpacking (`03_dataclass_unpacking.py`)
Automatic field extraction:
- Frozen dataclass unpacking
- Pydantic BaseModel support
- Multiple dataclasses
- Nested structures (recursive)

**Concepts:** dataclasses, Pydantic, field extraction, nested configs

---

### 4. Track Assets (`04_track_assets.py`)
ML model storage:
- Storing models as assets
- Hash-based deduplication
- Retrieving models
- Using hashes for efficiency

**Concepts:** assets, model storage, deduplication, get()

**Requirements:** `pip install scikit-learn`

---

### 5. Function Caching (`05_function_caching.py`)
Transparent result caching:
- Enabling result caching
- Cache hits and misses
- Cache lineage tracking
- Different params create different caches
- Hashable vs unhashable arguments

**Concepts:** cache, lineage, performance optimization

**Requirements:** `pip install xetrack[cache]`

---

### 6. Logging Integration (`06_logging_integration.py`)
Structured logging:
- Logging to stdout
- Logging to files
- JSONL format for ML datasets
- Logger-only mode (no database)
- Reading logs back

**Concepts:** logs, JSONL, model monitoring, log files

---

### 7. Data Analysis (`07_data_analysis.py`)
Pandas integration:
- Converting to DataFrame
- Pandas-like operations
- Filtering, grouping, aggregation
- Head/tail operations
- Statistical summaries

**Concepts:** DataFrame, pandas, analysis, Reader

---

### 8. SQL Queries (`08_sql_queries.py`)
Direct database queries:
- SQL with SQLite engine
- SQL with DuckDB engine
- Complex queries and aggregations
- Window functions (DuckDB)
- Analytics queries

**Concepts:** SQL, SQLite, DuckDB, aggregations

**Requirements (optional):** `pip install xetrack[duckdb]`

---

### 9. Model Monitoring (`09_model_monitoring.py`)
Production monitoring:
- Logger-only mode for production
- Real-time inference monitoring
- Drift detection preparation
- High-frequency logging
- Structured logging for analytics

**Concepts:** production, monitoring, inference logging, drift detection

---

### 10. Dataclass Tracking (Detailed) (`dataclass_tracking.py`)
**Original detailed dataclass example** - comprehensive demonstration of dataclass/Pydantic features.

---

## Data Organization

All examples store data in `examples_data/`:
```
examples_data/
├── quickstart.db          # Example 1
├── functions.db           # Example 2
├── dataclass.db           # Example 3
├── assets.db              # Example 4
├── cache.db               # Example 5
├── cache_dir/             # Example 5 cache storage
├── logging.db             # Example 6
├── logs/                  # Example 6 log files
├── analysis.db            # Example 7
├── sql_demo.db            # Example 8
└── monitoring/            # Example 9 monitoring logs
```

## Running Examples

### Run All Examples
```bash
cd xetrack
python examples/run_all.py
```

### Run Specific Example
```bash
python examples/01_quickstart.py
```

### Clean Up Data
```bash
rm -rf examples_data/
```

## Requirements

**Core examples** (1-3, 6-9):
- Only require base xetrack installation
- `pip install xetrack`

**Optional dependencies:**

- **Example 4** (Assets): `pip install xetrack[assets]` or `pip install scikit-learn`
- **Example 5** (Caching): `pip install xetrack[cache]`
- **Example 8** (DuckDB): `pip install xetrack[duckdb]`

Install all features:
```bash
pip install xetrack[assets,cache,duckdb]
```

## Tips

- Examples are self-contained and can run independently
- Each example cleans up after itself (mostly)
- Check `examples_data/` for generated databases and logs
- Use examples as templates for your own projects

## Troubleshooting

**Import errors:**
- Make sure xetrack is installed: `pip install xetrack`
- Install optional dependencies as needed

**Permission errors:**
- Ensure write access to `examples_data/` directory
- On first run, the directory will be created automatically

**Database locked:**
- Close any database viewers (DBeaver, SQLite Browser, etc.)
- Delete the specific `.db` file and run again

## Contributing

Found an issue or want to add an example?
- Report issues: https://github.com/xdssio/xetrack/issues
- Submit examples: PRs welcome!
