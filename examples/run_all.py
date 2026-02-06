"""
Run all xetrack examples

This script runs all examples in sequence, demonstrating
every feature of xetrack.
"""

import sys
import importlib.util
from pathlib import Path


def run_example(example_path: Path):
    """Run a single example file"""
    print("\n" + "=" * 70)
    print(f"Running: {example_path.name}")
    print("=" * 70 + "\n")
    
    try:
        # Import and run the example
        spec = importlib.util.spec_from_file_location("example", example_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, 'main'):
                module.main()
            
            return True
    except Exception as e:
        print(f"\n❌ Error running {example_path.name}: {e}")
        return False


def main():
    print("=" * 70)
    print("Xetrack Examples - Running All Examples")
    print("=" * 70)
    
    # Get examples directory
    examples_dir = Path(__file__).parent
    
    # Find all example files (numbered)
    example_files = sorted(examples_dir.glob("[0-9][0-9]_*.py"))
    
    if not example_files:
        print("\n❌ No example files found!")
        sys.exit(1)
    
    print(f"\nFound {len(example_files)} examples:")
    for ex in example_files:
        print(f"  - {ex.name}")
    
    print("\n" + "=" * 70)
    print("Starting examples...")
    print("=" * 70)
    
    # Run each example
    success_count = 0
    failed = []
    
    for example_file in example_files:
        if run_example(example_file):
            success_count += 1
        else:
            failed.append(example_file.name)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total examples: {len(example_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed examples:")
        for name in failed:
            print(f"  ❌ {name}")
    else:
        print("\n✅ All examples completed successfully!")
    
    print("\n" + "=" * 70)
    print("Examples complete!")
    print("Check examples_data/ for generated databases and logs")
    print("=" * 70)


if __name__ == "__main__":
    main()
