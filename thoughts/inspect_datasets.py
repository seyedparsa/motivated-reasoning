import datasets
from datasets import load_from_disk

def inspect_datasets():
    """Load and inspect the structure of the downloaded datasets."""
    
    print("=== MMLU Dataset Structure ===")
    try:
        mmlu = load_from_disk("data/mmlu")
        print(f"Splits available: {list(mmlu.keys())}")
        
        # Look at test split
        test_split = mmlu['test']
        print(f"Test split size: {len(test_split)}")
        print(f"Columns: {test_split.column_names}")
        
        # Show a sample
        print("\nSample MMLU question:")
        sample = test_split[0]
        for key, value in sample.items():
            print(f"  {key}: {value}")
        print()
        
    except Exception as e:
        print(f"Error loading MMLU: {e}")
    
    print("=== GPQA Dataset Structure ===")
    try:
        gpqa = load_from_disk("data/gpqa")
        print(f"Splits available: {list(gpqa.keys())}")
        
        # Look at train split (which is actually the test data in GPQA)
        train_split = gpqa['train']
        print(f"Train split size: {len(train_split)}")
        print(f"Columns: {train_split.column_names}")
        
        # Show a simplified sample (just the key fields)
        print("\nSample GPQA question (key fields only):")
        sample = train_split[0]
        key_fields = ['Question', 'Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3']
        for field in key_fields:
            if field in sample:
                print(f"  {field}: {sample[field]}")
        print()
        
    except Exception as e:
        print(f"Error loading GPQA: {e}")

if __name__ == "__main__":
    inspect_datasets()
