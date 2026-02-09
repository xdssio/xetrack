# DVC Setup Guide

Complete setup guide for DVC cache, remotes (S3), and worktree-shared configuration.

## Initial DVC Setup

```bash
# Initialize DVC in an existing git repo
cd your-project
dvc init
git add .dvc .dvcignore
git commit -m "init: DVC"
```

---

## DVC Cache Configuration

### Default Cache (Single Repo)

```bash
# Default location: .dvc/cache (inside repo)
# This is fine for single-repo, single-user setups
dvc config cache.type symlink,hardlink  # Avoid file duplication
```

### Shared Cache (Required for Worktrees)

When using `git worktree`, all worktrees share the same `.git` directory but each gets its own `.dvc/` directory. Without a shared cache, each worktree duplicates all DVC-tracked data.

```bash
# Set shared cache location (absolute path)
dvc cache dir /path/to/shared/dvc-cache

# Enable group permissions for team sharing
dvc config cache.shared group

# Use symlinks to avoid duplication
dvc config cache.type symlink,hardlink
```

For personal projects, a good shared cache location:
```bash
dvc cache dir ~/.dvc/shared-cache
```

For team/server setups:
```bash
dvc cache dir /data/dvc-cache
dvc config cache.shared group
```

### Worktree-Specific Setup

Each time you create a worktree, configure its DVC to use the shared cache:

```bash
git worktree add ../my-experiment -b exp/my-experiment
cd ../my-experiment

# Point this worktree's DVC to the shared cache
dvc cache dir /path/to/shared/dvc-cache

# Verify
dvc config cache.dir
```

You can automate this with a script (see `scripts/setup_worktree.sh` pattern):

```bash
#!/bin/bash
# create_worktree.sh <name>
SHARED_CACHE="/path/to/shared/dvc-cache"
git worktree add "../$1" -b "exp/$1"
cd "../$1"
dvc cache dir "$SHARED_CACHE"
echo "Worktree $1 created with shared DVC cache"
```

---

## Remote Storage (S3)

### Basic S3 Setup

```bash
# Add S3 remote as default
dvc remote add -d myremote s3://my-bucket/dvc-store

# Verify
dvc remote list
```

### S3 Authentication

Option 1: AWS CLI profile (recommended)
```bash
# Use default AWS CLI credentials
aws configure
# DVC will use these automatically
```

Option 2: Explicit credentials in DVC config
```bash
dvc remote modify myremote access_key_id YOUR_ACCESS_KEY
dvc remote modify myremote secret_access_key YOUR_SECRET_KEY
dvc remote modify myremote region us-east-1
```

Option 3: IAM role (for EC2/ECS/Lambda)
```bash
# No additional config needed, DVC uses instance role automatically
dvc remote modify myremote region us-east-1
```

Option 4: Environment variables
```bash
export AWS_ACCESS_KEY_ID=YOUR_KEY
export AWS_SECRET_ACCESS_KEY=YOUR_SECRET
export AWS_DEFAULT_REGION=us-east-1
```

### S3 Configuration Options

```bash
# Custom endpoint (MinIO, DigitalOcean Spaces, etc.)
dvc remote modify myremote endpointurl https://minio.example.com

# Server-side encryption
dvc remote modify myremote sse AES256

# Custom ACL
dvc remote modify myremote acl bucket-owner-full-control

# Increase transfer concurrency for large datasets
dvc remote modify myremote jobs 8
```

### Multiple Remotes

```bash
# Production remote (main storage)
dvc remote add -d production s3://prod-bucket/dvc-store

# Development remote (faster, cheaper)
dvc remote add dev s3://dev-bucket/dvc-store

# Push to specific remote
dvc push -r dev
dvc pull -r production
```

---

## Other Remote Types

### Google Cloud Storage
```bash
dvc remote add -d myremote gs://my-bucket/dvc-store
```

### Azure Blob Storage
```bash
dvc remote add -d myremote azure://my-container/dvc-store
dvc remote modify myremote connection_string "DefaultEndpointsProtocol=..."
```

### Local/Network Path
```bash
dvc remote add -d myremote /mnt/nfs/dvc-store
```

### SSH
```bash
dvc remote add -d myremote ssh://user@server:/path/to/store
```

---

## DVC Push/Pull Workflow

### Push (after experiment)
```bash
# Push all DVC-tracked files to remote
dvc push

# Push specific file
dvc push data/train.parquet.dvc

# Push with verbose output
dvc push -v
```

### Pull (restore experiment)
```bash
# Pull all DVC-tracked files
dvc pull

# Pull from specific tag
git checkout e0.1.0
dvc pull

# Pull specific file
dvc pull models/production/model.bin.dvc
```

### Checkout (switch DVC state without downloading)
```bash
# After git checkout, sync DVC to match
dvc checkout

# This uses the local cache, no network needed
# If cache miss, use dvc pull
```

---

## DVC Pipeline Setup

### dvc.yaml Template

```yaml
stages:
  train:
    cmd: python train.py --config params.yaml
    deps:
      - train.py
      - data/train.parquet
    params:
      - model.name
      - model.learning_rate
      - model.batch_size
    outs:
      - models/production/model.bin
    metrics:
      - results/metrics.json:
          cache: false

  merge_results:
    cmd: python scripts/merge_artifacts.py
      --strategy ${merge.strategy}
      --base-db results/experiments.db
      --exp-db results/incoming.db
      --key-columns ${merge.key_columns}
    deps:
      - scripts/merge_artifacts.py
      - results/incoming.db
    params:
      - merge.strategy
      - merge.key_columns
    outs:
      - results/experiments.db:
          persist: true
```

### params.yaml Template

```yaml
model:
  name: bert-base
  learning_rate: 0.0001
  batch_size: 32

merge:
  strategy: merge
  key_columns:
    - run_id
    - version
```

### Running the Pipeline

```bash
# Run all stages
dvc repro

# Run specific stage
dvc repro merge_results

# See the DAG
dvc dag

# Check what would run (dry run)
dvc repro --dry
```

---

## What to Track with DVC

### Track with DVC (large/binary files)
- `data/*.parquet` - training and test data
- `results/experiments.db` - SQLite database
- `models/production/model.bin` - production model
- `models/candidates/` - candidate model directory

### Track with Git (small/text files)
- `*.py` - all code
- `dvc.yaml` - pipeline definition
- `dvc.lock` - pipeline state (auto-generated, MUST be committed)
- `params.yaml` - parameters
- `*.dvc` - DVC tracking files (auto-generated)
- `.dvc/config` - DVC configuration

### The .gitignore Pattern

When you `dvc add data/train.parquet`, DVC automatically:
1. Creates `data/train.parquet.dvc` (git-tracked pointer)
2. Adds `data/train.parquet` to `data/.gitignore`
3. Moves the actual file to `.dvc/cache/`
4. Creates a symlink (if configured)

---

## Verification Commands

```bash
# Check DVC status
dvc status

# Check remote connectivity
dvc remote list
dvc push --dry  # Dry run push

# Verify cache
dvc cache dir
ls $(dvc cache dir)

# Check what's tracked
dvc list . --dvc-only

# Garbage collect unused cache
dvc gc --workspace  # Keep only current workspace
dvc gc --all-tags   # Keep all tagged versions
```

---

## Troubleshooting

**"Cache miss" errors after worktree switch:**
- Ensure shared cache is configured: `dvc cache dir`
- Run `dvc pull` to fetch from remote

**"Database is locked" with DuckDB:**
- Only one process can write to a DuckDB file DB at a time
- Use separate DB files per process, merge afterward
- Or use SQLite with xetrack for single-writer scenarios

**Large push/pull times:**
- Increase concurrency: `dvc remote modify myremote jobs 8`
- Use `dvc push/pull -j 4` for parallel transfers
- Consider chunking large datasets into multiple parquet files

**"Modified" status after DVC checkout:**
- Run `dvc checkout` after `git checkout`
- If persists: `dvc checkout --force`
