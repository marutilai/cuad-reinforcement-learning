# cuad_qa/data/convert_cuad_dataset.py

import os
import json
import argparse
import logging
from tqdm import tqdm
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, ValidationError
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import login

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- Pydantic Models ---
class AnswerSpan(BaseModel):
    """Represents a single text span answer for a clause."""

    text: str
    answer_start: int  # Character offset within the PARENT PARAGRAPH's context

    @property
    def answer_end(self) -> int:
        """Calculate end character offset."""
        return self.answer_start + len(self.text)


class QuestionAnswer(BaseModel):
    """Represents a question (clause type) and its associated answers for a paragraph."""

    question: str = Field(description="The question identifying the clause type.")
    qa_id: str = Field(
        alias="id",
        description="Original QA identifier from CUAD.",
        validation_alias="id",
    )
    answers: List[AnswerSpan] = Field(
        default_factory=list,
        description="List of text spans identifying the clause within the paragraph.",
    )
    is_impossible: bool = Field(
        default=False,
        description="True if this clause type is explicitly marked as not present in the paragraph.",
    )

    class Config:
        populate_by_name = True


class Paragraph(BaseModel):
    """Represents a single paragraph within a contract."""

    context: str = Field(description="The text content of the paragraph.")
    qas: List[QuestionAnswer] = Field(
        default_factory=list,
        description="Questions and answers relevant to this paragraph.",
    )


class Contract(BaseModel):
    """Represents a single contract document with its paragraphs and annotations."""

    title: str = Field(description="The title or filename of the contract.")
    paragraphs: List[Paragraph] = Field(
        description="List of paragraphs within the contract."
    )

    @property
    def full_text(self) -> str:
        """Concatenates all paragraph contexts to get the full contract text."""
        separator = "\n\n--- PARAGRAPH BREAK ---\n\n"
        return separator.join([p.context for p in self.paragraphs])

    def get_annotations_with_full_text_offsets(self) -> List[Dict[str, Any]]:
        """Generates a flat list of annotations with offsets relative to the full contract text."""
        annotations = []
        current_offset = 0
        separator = "\n\n--- PARAGRAPH BREAK ---\n\n"

        for para_idx, p in enumerate(self.paragraphs):
            for qa in p.qas:
                if not qa.is_impossible and qa.answers:
                    for answer in qa.answers:
                        start_in_full = current_offset + answer.answer_start
                        end_in_full = start_in_full + len(answer.text)

                        annotations.append(
                            {
                                "contract_title": self.title,
                                "clause_type": qa.question,
                                "text": answer.text,
                                "start_char_in_full_text": start_in_full,
                                "end_char_in_full_text": end_in_full,
                            }
                        )
            current_offset += len(p.context) + len(separator)

        return annotations


def load_cuad_data(data_path: str) -> List[Dict[str, Any]]:
    """Load the CUAD dataset from the main JSON file."""
    logging.info(f"Loading CUAD data from {data_path}...")
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "data" not in data:
            raise ValueError("Invalid CUAD format: 'data' key not found.")
        logging.info(f"Loaded {len(data['data'])} raw contract entries from JSON.")
        return data["data"]
    except FileNotFoundError:
        logging.error(f"Error: Data file not found at {data_path}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {data_path}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred loading data: {e}")
        raise


def parse_contracts(
    raw_contracts_data: List[Dict[str, Any]], max_contracts: Optional[int] = None
) -> List[Contract]:
    """Parses raw contract data into a list of Pydantic Contract objects."""
    if max_contracts and max_contracts < len(raw_contracts_data):
        logging.info(f"Limiting processing to {max_contracts} contracts.")
        raw_contracts_data = raw_contracts_data[:max_contracts]

    parsed_contracts: List[Contract] = []
    logging.info("Parsing raw contract data into Pydantic models...")

    for raw_contract in tqdm(raw_contracts_data, desc="Parsing Contracts"):
        try:
            paragraphs_data = []
            raw_paragraphs = raw_contract.get("paragraphs", [])
            if not raw_paragraphs:
                logging.warning(
                    f"Contract '{raw_contract.get('title', 'N/A')}' has no paragraphs. Skipping."
                )
                continue

            for para_idx, raw_para in enumerate(raw_paragraphs):
                raw_para["original_index"] = para_idx
                if "context" not in raw_para:
                    logging.warning(
                        f"Paragraph {para_idx} in contract '{raw_contract.get('title', 'N/A')}' missing 'context'. Skipping paragraph."
                    )
                    continue

                validated_qas = []
                for raw_qa in raw_para.get("qas", []):
                    try:
                        # Add debug logging
                        if "id" not in raw_qa:
                            logging.debug(f"QA entry missing 'id' field: {raw_qa}")
                        validated_qa = QuestionAnswer.model_validate(raw_qa)
                        validated_qas.append(validated_qa)
                    except ValidationError as qa_err:
                        logging.warning(
                            f"Skipping invalid QA in paragraph {para_idx} of contract '{raw_contract.get('title', 'N/A')}': {qa_err}"
                        )
                        # Add debug print of the failing QA
                        logging.debug(f"Failed QA content: {raw_qa}")

                raw_para["qas"] = validated_qas
                paragraphs_data.append(Paragraph.model_validate(raw_para))

            raw_contract["paragraphs"] = paragraphs_data
            contract_obj = Contract.model_validate(raw_contract)
            parsed_contracts.append(contract_obj)

        except ValidationError as e:
            logging.error(
                f"Failed to validate contract '{raw_contract.get('title', 'N/A')}': {e}. Skipping contract."
            )
        except Exception as e:
            logging.error(
                f"Unexpected error processing contract '{raw_contract.get('title', 'N/A')}': {e}. Skipping contract."
            )

    logging.info(
        f"Successfully parsed {len(parsed_contracts)} contracts into Pydantic objects."
    )
    return parsed_contracts


def save_to_disk(contracts: List[Contract], input_file: str):
    """Save the structured contracts to disk as JSON in the ./data subdirectory relative to project root."""
    # Assuming this script is in cuad_qa/data/
    # Output to cuad_qa/data/processed_CUADv1.json (or similar)
    output_dir = os.path.dirname(
        os.path.abspath(__file__)
    )  # This script's directory (cuad_qa/data)
    base_input_filename = os.path.basename(input_file)
    # Construct output filename like processed_CUADv1.json in the same dir as this script
    output_filename = f"processed_{base_input_filename}"  # Or a fixed name like "processed_cuad_contracts.json"
    output_file = os.path.join(output_dir, output_filename)

    logging.info(f"Saving processed data to {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        json_data = [contract.model_dump(mode="json") for contract in contracts]
        # The schema {"version": "cuad_v1.0", "data": json_data} is good if load_processed_data expects it
        json.dump({"version": "cuad_v1.0", "data": json_data}, f, indent=2)

    logging.info(f"Saved processed data to {output_file}")


def upload_to_huggingface(contracts: List[Contract], repo_id: str):
    """Upload the structured contracts to HuggingFace as a DatasetDict"""
    # Load environment variables
    load_dotenv()

    # Get token and login
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        logging.error("HUGGINGFACE_TOKEN not found in environment variables")
        raise ValueError("Please set HUGGINGFACE_TOKEN in .env file")

    login(token)
    logging.info(f"Preparing dataset for upload to {repo_id}...")

    # Create flattened annotations list
    all_annotations = []
    for contract in contracts:
        all_annotations.extend(contract.get_annotations_with_full_text_offsets())

    # Prepare datasets
    contracts_dict = {
        "title": [c.title for c in contracts],
        "full_text": [c.full_text for c in contracts],
    }

    annotations_dict = {
        "contract_title": [a["contract_title"] for a in all_annotations],
        "clause_type": [a["clause_type"] for a in all_annotations],
        "text": [a["text"] for a in all_annotations],
        "start_char": [a["start_char_in_full_text"] for a in all_annotations],
        "end_char": [a["end_char_in_full_text"] for a in all_annotations],
    }

    # Create datasets
    contracts_ds = Dataset.from_dict(contracts_dict)
    annotations_ds = Dataset.from_dict(annotations_dict)

    # Push to Hugging Face
    logging.info(f"Uploading datasets to Hugging Face ({repo_id})...")
    try:
        contracts_ds.push_to_hub(f"{repo_id}/contracts", private=True)
        annotations_ds.push_to_hub(f"{repo_id}/annotations", private=True)
        logging.info(
            f"Datasets uploaded successfully to https://huggingface.co/datasets/{repo_id}"
        )
    except Exception as e:
        logging.error(f"Failed to upload to Hugging Face: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Process CUAD contract dataset")
    parser.add_argument(
        "--max-contracts",
        type=int,
        default=None,
        help="Maximum number of contracts to process (default: all)",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="raw_data/CUADv1.json",
        help="Path to the CUAD JSON file (default: raw_data/CUADv1.json)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to HuggingFace (requires --repo-id)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        help="HuggingFace repository ID for upload (e.g., 'username/cuad-processed')",
    )
    args = parser.parse_args()

    if args.upload and not args.repo_id:
        parser.error("--upload requires --repo-id")

    try:
        # Load and parse contracts
        raw_contracts_data = load_cuad_data(args.data_file)
        parsed_contracts = parse_contracts(
            raw_contracts_data,
            max_contracts=args.max_contracts,
        )

        if not parsed_contracts:
            logging.warning("No contracts were successfully parsed. Exiting.")
            return

        # Save to disk in the same directory as input file
        save_to_disk(parsed_contracts, args.data_file)

        # Upload to HuggingFace if requested
        if args.upload:
            upload_to_huggingface(parsed_contracts, args.repo_id)

        logging.info(
            f"Processing complete. {len(parsed_contracts)} contracts processed."
        )

    except Exception as e:
        logging.error(f"Script failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
