# cuad_qa/data/generate_cuad_scenarios.py

import os
import sqlite3
import random
import logging
import argparse
from typing import List, Dict, Any, Tuple

from pydantic import BaseModel, Field
from datasets import Dataset, DatasetDict, Features, Value, Sequence
from huggingface_hub import create_repo
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables (optional, for HF token etc.)
load_dotenv()

# Use the DB path defined in the DB creation script
from .local_contract_db import DEFAULT_DB_PATH as DEFAULT_CUAD_DB_PATH

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- Pydantic Model for Training Scenarios ---
class ClauseFindingScenario(BaseModel):
    """Defines a single training/evaluation task for the agent."""

    scenario_id: int = Field(description="Unique identifier for this scenario.")
    contract_id: int = Field(description="Primary key of the contract in the DB.")
    contract_title: str = Field(description="Title of the contract.")
    annotation_id: int = Field(
        description="Primary key of the specific annotation in the DB."
    )
    clause_type: str = Field(
        description="The type of clause to find (from CUAD question)."
    )
    ground_truth_clause_text: str = Field(
        description="The exact ground truth text of the clause."
    )
    start_char: int = Field(
        description="Start character offset in the full contract text."
    )
    end_char: int = Field(description="End character offset in the full contract text.")
    # We don't need 'how_realistic' or 'query_date' from the original email script


# --- Database Interaction ---
def fetch_annotations_and_contracts(
    db_path: str = DEFAULT_CUAD_DB_PATH,
) -> List[Dict[str, Any]]:
    """Fetches all annotations joined with their contract titles from the DB."""
    logging.info(f"Fetching annotations from database: {db_path}")
    if not os.path.exists(db_path):
        logging.error(
            f"Database file not found at {db_path}. Run local_contract_db.py first."
        )
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)  # Read-only connection
    cursor = conn.cursor()
    query = """
        SELECT
            a.id as annotation_id,
            a.contract_id,
            c.title as contract_title,
            a.clause_type,
            a.text as ground_truth_clause_text,
            a.start_char,
            a.end_char
        FROM annotations a
        JOIN contracts c ON a.contract_id = c.id;
    """
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        # Convert rows to list of dictionaries
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in rows]
        logging.info(f"Fetched {len(results)} annotations from the database.")
    except sqlite3.Error as e:
        logging.error(f"Database query failed: {e}")
        results = []
    finally:
        conn.close()
    return results


# --- Scenario Creation and Splitting ---
def create_scenarios(
    annotation_data: List[Dict[str, Any]],
) -> List[ClauseFindingScenario]:
    """Converts fetched annotation data into ClauseFindingScenario objects."""
    scenarios = []
    logging.info("Creating scenario objects from annotation data...")
    for i, data in enumerate(tqdm(annotation_data, desc="Creating Scenarios")):
        try:
            scenario = ClauseFindingScenario(
                scenario_id=data[
                    "annotation_id"
                ],  # Use DB annotation ID as scenario ID for traceability
                contract_id=data["contract_id"],
                contract_title=data["contract_title"],
                annotation_id=data["annotation_id"],
                clause_type=data["clause_type"],
                ground_truth_clause_text=data["ground_truth_clause_text"],
                start_char=data["start_char"],
                end_char=data["end_char"],
            )
            scenarios.append(scenario)
        except Exception as e:  # Catch potential validation or key errors
            logging.warning(f"Skipping annotation due to error: {e}. Data: {data}")
    logging.info(f"Created {len(scenarios)} scenarios.")
    return scenarios


def split_scenarios_by_contract(
    scenarios: List[ClauseFindingScenario], train_ratio: float = 0.8, seed: int = 42
) -> Tuple[List[ClauseFindingScenario], List[ClauseFindingScenario]]:
    """Splits scenarios into train/test sets based on contract ID."""
    logging.info(f"Splitting scenarios by contract_id (Train ratio: {train_ratio})...")
    contracts_by_id: Dict[int, List[ClauseFindingScenario]] = {}
    for s in scenarios:
        if s.contract_id not in contracts_by_id:
            contracts_by_id[s.contract_id] = []
        contracts_by_id[s.contract_id].append(s)

    unique_contract_ids = list(contracts_by_id.keys())
    random.seed(seed)
    random.shuffle(unique_contract_ids)

    num_train = int(len(unique_contract_ids) * train_ratio)
    train_contract_ids = set(unique_contract_ids[:num_train])
    test_contract_ids = set(unique_contract_ids[num_train:])

    train_scenarios: List[ClauseFindingScenario] = []
    test_scenarios: List[ClauseFindingScenario] = []

    for contract_id in train_contract_ids:
        train_scenarios.extend(contracts_by_id[contract_id])
    for contract_id in test_contract_ids:
        test_scenarios.extend(contracts_by_id[contract_id])

    logging.info(
        f"Split complete: {len(train_scenarios)} train scenarios, {len(test_scenarios)} test scenarios."
    )
    return train_scenarios, test_scenarios


# --- Hugging Face Dataset Creation ---
def create_and_push_dataset(
    train_scenarios: List[ClauseFindingScenario],
    test_scenarios: List[ClauseFindingScenario],
    hf_repo_id: str,
):
    """Creates a DatasetDict and pushes it to the Hugging Face Hub."""
    logging.info(f"Creating Hugging Face dataset for repo: {hf_repo_id}")

    # Define the features based on the ClauseFindingScenario model
    features = Features(
        {
            "scenario_id": Value("int32"),
            "contract_id": Value("int32"),
            "contract_title": Value("string"),
            "annotation_id": Value("int32"),
            "clause_type": Value("string"),
            "ground_truth_clause_text": Value("string"),
            "start_char": Value("int32"),
            "end_char": Value("int32"),
        }
    )

    # Helper function to convert list of Pydantic models to dict for Dataset
    def to_dict(scenario_list: List[ClauseFindingScenario]) -> Dict[str, Any]:
        return {
            "scenario_id": [s.scenario_id for s in scenario_list],
            "contract_id": [s.contract_id for s in scenario_list],
            "contract_title": [s.contract_title for s in scenario_list],
            "annotation_id": [s.annotation_id for s in scenario_list],
            "clause_type": [s.clause_type for s in scenario_list],
            "ground_truth_clause_text": [
                s.ground_truth_clause_text for s in scenario_list
            ],
            "start_char": [s.start_char for s in scenario_list],
            "end_char": [s.end_char for s in scenario_list],
        }

    # Create Dataset objects
    train_dataset = Dataset.from_dict(to_dict(train_scenarios), features=features)
    test_dataset = Dataset.from_dict(to_dict(test_scenarios), features=features)

    # Create DatasetDict
    dataset_dict = DatasetDict({"train": train_dataset, "test": test_dataset})

    logging.info(f"Dataset structure:\n{dataset_dict}")

    # Create repo and push the dataset
    try:
        logging.info(f"Creating repository '{hf_repo_id}' on Hugging Face Hub...")
        create_repo(hf_repo_id, repo_type="dataset", exist_ok=True)
        logging.info(f"Pushing dataset to {hf_repo_id}...")
        dataset_dict.push_to_hub(hf_repo_id, private=False)  # Adjust private as needed
        logging.info(
            f"Dataset successfully pushed to https://huggingface.co/datasets/{hf_repo_id}"
        )
    except Exception as e:
        logging.error(f"Failed to push dataset to Hugging Face Hub: {e}")
        logging.error(
            "Ensure you are logged in (`huggingface-cli login`) and have permissions."
        )
        raise


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description="Generate CUAD training scenarios from DB"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=DEFAULT_CUAD_DB_PATH,
        help=f"Path to the CUAD SQLite DB file (default: {DEFAULT_CUAD_DB_PATH})",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        required=True,
        help="HuggingFace repository ID to upload the scenario dataset (e.g., 'username/cuad-scenarios')",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of contracts to use for the training set (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test splitting (default: 42)",
    )
    args = parser.parse_args()

    try:
        # 1. Fetch data from DB
        annotation_data = fetch_annotations_and_contracts(args.db_path)
        if not annotation_data:
            logging.error("No annotations fetched from the database. Exiting.")
            return

        # 2. Create scenario objects
        scenarios = create_scenarios(annotation_data)
        if not scenarios:
            logging.error("No scenarios created from annotations. Exiting.")
            return

        # 3. Split into train/test
        train_scenarios, test_scenarios = split_scenarios_by_contract(
            scenarios, train_ratio=args.train_ratio, seed=args.seed
        )

        # 4. Create and push Hugging Face dataset
        create_and_push_dataset(train_scenarios, test_scenarios, args.hf_repo_id)

        logging.info("Scenario generation and upload process completed successfully.")

    except FileNotFoundError:
        # Specific handling for DB not found
        logging.error(
            "Database file required but not found. Please run the database creation script first."
        )
        exit(1)
    except Exception as e:
        logging.error(f"Script failed with an unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
