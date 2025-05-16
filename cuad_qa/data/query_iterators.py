# cuad_qa/data/query_iterators.py

import random
import logging
from typing import List, Optional
from datasets import load_dataset, Dataset
from .generate_cuad_scenarios import ClauseFindingScenario

import numpy as np

# Define the Hugging Face repository ID for CUAD scenarios
HF_REPO_ID = "marutiagarwal/cuad-qa-scenarios"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def convert_ndarray_to_list(d):
    # Recursively convert all numpy arrays to lists/ints/floats
    if isinstance(d, dict):
        return {k: convert_ndarray_to_list(v) for k, v in d.items()}
    elif isinstance(d, np.ndarray):
        # Flatten to scalar if array is scalar
        if d.shape == ():
            return d.item()
        return d.tolist()
    else:
        return d


def load_clause_finding_scenarios(
    split: str = "train",
    limit: Optional[int] = None,
    shuffle: bool = False,
) -> List[ClauseFindingScenario]:
    """
    Loads Clause Finding Scenarios from the specified Hugging Face dataset and makes them available
    for the training loop (train.py) when it calls this function.

    Args:
        split: The dataset split to load ('train' or 'test').
        limit: Maximum number of scenarios to return.
        shuffle: Whether to shuffle the dataset before limiting.

    Returns:
        A list of ClauseFindingScenario objects.
    """
    logging.info(f"Loading CUAD scenarios from '{HF_REPO_ID}' (split: {split})")
    try:
        # Load the dataset split
        # cache_dir can be added if needed: cache_dir="path/to/cache"
        dataset: Dataset = load_dataset(HF_REPO_ID, split=split)  # type: ignore
    except Exception as e:
        logging.error(f"Failed to load dataset '{HF_REPO_ID}' split '{split}': {e}")
        logging.error(
            "Please ensure the dataset exists, is public or you are logged in (`huggingface-cli login`), and you have internet access."
        )
        raise  # Re-raise exception to signal failure

    # Shuffle if requested
    if shuffle:
        logging.info("Shuffling dataset...")
        dataset = dataset.shuffle(seed=42)  # Added seed for reproducibility

    # Convert rows to Pydantic objects
    scenarios: List[ClauseFindingScenario] = []
    logging.info(f"Converting {len(dataset)} rows to ClauseFindingScenario objects...")
    skipped_count = 0
    for row in dataset:
        try:
            row = convert_ndarray_to_list(row)
            scenarios.append(ClauseFindingScenario(**row))
        except Exception as e:
            logging.warning(f"Skipping row due to conversion error: {e}. Row: {row}")
            skipped_count += 1

    if skipped_count > 0:
        logging.warning(f"Skipped {skipped_count} rows during Pydantic conversion.")
    logging.info(f"Successfully created {len(scenarios)} scenario objects.")

    # Apply limit if specified
    if limit is not None:
        actual_limit = min(limit, len(scenarios))
        logging.info(f"Applying limit: returning {actual_limit} scenarios.")
        return scenarios[:actual_limit]
    else:
        return scenarios


# Example usage block (optional, useful for testing the script directly)
if __name__ == "__main__":
    print("--- Testing query_iterator ---")
    try:
        print("\nTesting loading train split...")
        # Reduced limit for faster testing
        train_scenarios = load_clause_finding_scenarios(
            split="train", limit=5, shuffle=True
        )
        print(f"Loaded {len(train_scenarios)} train scenarios.")
        if train_scenarios:
            print("First train scenario example:")
            print(train_scenarios[0].model_dump_json(indent=2))  # Pretty print

        print("\nTesting loading test split...")
        test_scenarios = load_clause_finding_scenarios(split="test", limit=5)
        print(f"Loaded {len(test_scenarios)} test scenarios.")
        if test_scenarios:
            print("First test scenario example:")
            print(test_scenarios[0].model_dump_json(indent=2))  # Pretty print

        print("\n--- Test complete ---")

    except Exception as e:
        print(f"\nError during example usage: {e}")
        print("Ensure the Hugging Face dataset exists and is accessible.")
