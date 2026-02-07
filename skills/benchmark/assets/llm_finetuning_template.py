"""
LLM Finetuning Simulation Template

Simulates LLM finetuning by tracking loss curves, learning rates, and metrics.
Useful for testing xetrack with ML training workflows without actual GPU compute.
"""

from dataclasses import dataclass
from typing import Optional
from xetrack import Tracker, Reader
import time
import math


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Training hyperparameters."""
    model_name: str  # 'gpt2', 'llama-7b', 'bert-base'
    learning_rate: float
    batch_size: int
    num_epochs: int
    warmup_steps: int = 100
    weight_decay: float = 0.01
    optimizer: str = 'adamw'


def simulate_training_step(step: int, config: TrainingConfig) -> dict:
    """
    Simulate a single training step.

    Returns metrics for this step:
    - loss (simulated decay)
    - learning_rate (with warmup)
    - gradient_norm
    - step_time
    """
    # Simulate realistic loss decay
    # Loss starts high and decreases with some noise
    base_loss = 3.5 * math.exp(-step / 500) + 0.5
    noise = 0.1 * math.sin(step * 0.1)  # Add some oscillation
    loss = max(0.1, base_loss + noise)

    # Learning rate with warmup
    if step < config.warmup_steps:
        lr = config.learning_rate * (step / config.warmup_steps)
    else:
        # Cosine decay after warmup
        progress = (step - config.warmup_steps) / max(1, 1000 - config.warmup_steps)
        lr = config.learning_rate * (0.5 * (1 + math.cos(math.pi * progress)))

    # Simulate gradient norm (decreases as model converges)
    grad_norm = 10.0 * math.exp(-step / 300) + 0.5

    # Simulate step time (batch size dependent)
    step_time = 0.01 + (config.batch_size / 32) * 0.001

    # Simulate actual training time
    time.sleep(step_time / 10)  # Speed up for demo

    return {
        'step': step,
        'loss': round(loss, 4),
        'learning_rate': round(lr, 6),
        'gradient_norm': round(grad_norm, 4),
        'step_time': round(step_time, 4),
        'error': None
    }


def simulate_evaluation(step: int, config: TrainingConfig) -> dict:
    """
    Simulate evaluation metrics on validation set.

    Returns:
    - val_loss
    - val_accuracy
    - val_perplexity
    """
    # Validation metrics improve over time
    val_loss = 2.0 * math.exp(-step / 600) + 0.7
    val_accuracy = min(0.95, 0.5 + 0.4 * (1 - math.exp(-step / 400)))
    val_perplexity = math.exp(val_loss)

    return {
        'step': step,
        'val_loss': round(val_loss, 4),
        'val_accuracy': round(val_accuracy, 4),
        'val_perplexity': round(val_perplexity, 4)
    }


def run_finetuning_benchmark():
    """
    Simulate finetuning multiple models with different hyperparameters.

    Tracks:
    - Training steps (loss, lr, gradients) → 'training_steps' table
    - Evaluation checkpoints → 'eval_checkpoints' table
    - Final metrics → 'final_metrics' table
    """

    # Define training configurations to benchmark
    configs = [
        TrainingConfig(model_name='gpt2', learning_rate=1e-4, batch_size=16, num_epochs=1),
        TrainingConfig(model_name='gpt2', learning_rate=5e-5, batch_size=32, num_epochs=1),
        TrainingConfig(model_name='bert-base', learning_rate=2e-5, batch_size=16, num_epochs=1),
    ]

    total_steps_per_run = 1000
    eval_every = 100

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Finetuning: {config.model_name}")
        print(f"LR: {config.learning_rate}, Batch: {config.batch_size}")
        print(f"{'='*60}")

        # Separate trackers for different data types
        steps_tracker = Tracker(
            db='llm_finetuning.db',
            engine='duckdb',
            cache='finetuning_cache',
            table='training_steps',
            params={'run_id': f"{config.model_name}-{config.learning_rate}-{config.batch_size}"}
        )

        eval_tracker = Tracker(
            db='llm_finetuning.db',
            engine='duckdb',
            table='eval_checkpoints',
            params={'run_id': f"{config.model_name}-{config.learning_rate}-{config.batch_size}"}
        )

        metrics_tracker = Tracker(
            db='llm_finetuning.db',
            engine='duckdb',
            table='final_metrics',
            params={'run_id': f"{config.model_name}-{config.learning_rate}-{config.batch_size}"}
        )

        # Simulate training loop
        best_val_loss = float('inf')
        for step in range(1, total_steps_per_run + 1):
            # Training step
            step_result = steps_tracker.track(
                simulate_training_step,
                args=[step, config]
            )

            # Periodic evaluation
            if step % eval_every == 0:
                eval_result = eval_tracker.track(
                    simulate_evaluation,
                    args=[step, config]
                )

                print(f"  Step {step:4d} | "
                      f"Loss: {step_result['loss']:.4f} | "
                      f"Val Loss: {eval_result['val_loss']:.4f} | "
                      f"Val Acc: {eval_result['val_accuracy']:.4f}")

                if eval_result['val_loss'] < best_val_loss:
                    best_val_loss = eval_result['val_loss']

        # Log final aggregated metrics
        final_step_result = steps_tracker.latest
        final_eval_result = eval_tracker.latest

        metrics_tracker.log({
            'model_name': config.model_name,
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'total_steps': total_steps_per_run,
            'final_train_loss': final_step_result['loss'],
            'final_val_loss': final_eval_result['val_loss'],
            'final_val_accuracy': final_eval_result['val_accuracy'],
            'best_val_loss': best_val_loss,
            'avg_step_time': steps_tracker.to_df()['step_time'].mean()
        })

    print("\n" + "="*60)
    print("✅ Finetuning benchmark complete!")
    print("\nAnalyze results:")
    print("  # Final metrics")
    print("  python -c \"from xetrack import Reader; print(Reader('llm_finetuning.db', table='final_metrics', engine='duckdb').to_df())\"")
    print("\n  # Training curves")
    print("  python -c \"from xetrack import Reader; df = Reader('llm_finetuning.db', table='training_steps', engine='duckdb').to_df(); print(df.groupby('params_run_id')[['loss', 'learning_rate']].describe())\"")


if __name__ == "__main__":
    run_finetuning_benchmark()
