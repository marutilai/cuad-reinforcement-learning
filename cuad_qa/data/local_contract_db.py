# cuad_qa/data/local_contract_db.py

import os, sys
import json
import logging
import sqlite3
from tqdm import tqdm
from typing import List, Dict, Any
from pydantic import ValidationError

# path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if path not in sys.path:
#     sys.path.append(path)

from cuad_qa.data.convert_cuad_dataset import Contract

# Assuming the previous script saved its output here:
DEFAULT_PROCESSED_JSON_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "processed_cuad_contracts.json"
)
# Database will live in "data/cuad_contracts.db" relative to project root
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "cuad_contracts.db"
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- Database Schema for Contracts and Annotations ---
SQL_CREATE_TABLES = """
DROP TABLE IF EXISTS annotations;
DROP TABLE IF EXISTS contracts_fts;
DROP TABLE IF EXISTS contracts;

CREATE TABLE contracts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT UNIQUE NOT NULL, -- Use title as a unique identifier for contracts
    full_text TEXT NOT NULL
);

CREATE TABLE annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    contract_id INTEGER NOT NULL,       -- Foreign key to contracts table
    clause_type TEXT NOT NULL,          -- The question identifying the clause
    text TEXT NOT NULL,                 -- The ground truth text of the clause
    start_char INTEGER NOT NULL,        -- Start offset IN FULL_TEXT
    end_char INTEGER NOT NULL,          -- End offset IN FULL_TEXT
    FOREIGN KEY(contract_id) REFERENCES contracts(id) ON DELETE CASCADE
);
"""

SQL_CREATE_INDEXES_TRIGGERS = """
-- Index on contract title for potentially faster lookups if needed elsewhere
CREATE INDEX idx_contracts_title ON contracts(title);

-- Indexes on annotations for potential filtering/grouping during scenario generation
CREATE INDEX idx_annotations_contract_id ON annotations(contract_id);
CREATE INDEX idx_annotations_clause_type ON annotations(clause_type);

-- Create FTS5 table indexing the full contract text
CREATE VIRTUAL TABLE contracts_fts USING fts5(
    title,      -- Also index the title for search flexibility
    full_text,
    content='contracts',
    content_rowid='id' -- Links to the contracts table's primary key
);

-- Triggers to keep FTS table synchronized with the contracts table
CREATE TRIGGER contracts_ai AFTER INSERT ON contracts BEGIN
    INSERT INTO contracts_fts (rowid, title, full_text)
    VALUES (new.id, new.title, new.full_text);
END;

CREATE TRIGGER contracts_ad AFTER DELETE ON contracts BEGIN
    DELETE FROM contracts_fts WHERE rowid=old.id;
END;

CREATE TRIGGER contracts_au AFTER UPDATE ON contracts BEGIN
    UPDATE contracts_fts SET title=new.title, full_text=new.full_text WHERE rowid=old.id;
END;

-- Optional: Populate FTS table immediately after creation if needed,
-- though triggers handle ongoing updates. Usually done AFTER bulk insert.
-- INSERT INTO contracts_fts (rowid, title, full_text) SELECT id, title, full_text FROM contracts;
"""


# --- Functions ---


def load_processed_data(json_path: str) -> List[Dict[str, Any]]:
    """Loads the processed contract data from the JSON file."""
    logging.info(f"Loading processed contract data from: {json_path}")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Expecting format: {"version": "...", "data": [list of contract dicts]}
        if "data" not in data or not isinstance(data["data"], list):
            raise ValueError(
                "Invalid processed JSON format. Expected a 'data' key with a list of contracts."
            )
        logging.info(f"Loaded {len(data['data'])} processed contract objects.")
        return data["data"]
    except FileNotFoundError:
        logging.error(f"Error: Processed data file not found at {json_path}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {json_path}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred loading processed data: {e}")
        raise


def create_database(db_path: str):
    """Creates the SQLite database and the defined tables."""
    logging.info(f"Creating SQLite database and tables at: {db_path}")
    # Ensure directory exists
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.executescript(SQL_CREATE_TABLES)
        conn.commit()
        logging.info("Database tables created successfully.")
    except sqlite3.Error as e:
        logging.error(f"Database table creation failed: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def populate_database(db_path: str, processed_contracts: List[Dict[str, Any]]):
    """Populates the database with contracts and their annotations."""
    logging.info(f"Populating database '{db_path}'...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # --- Performance Pragmas (less critical than FTS but can help bulk inserts) ---
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA journal_mode = MEMORY;")

    contract_count = 0
    annotation_count = 0
    contracts_to_insert = []
    annotations_to_insert = []
    # Need to recalculate annotations with full text offsets here
    # We need the Contract Pydantic model from the previous script for its helpers
    # Import it (assuming it's saved in a reachable path)
    try:
        # Adjust the import path based on your project structure
        from .convert_cuad_dataset import Contract
    except ImportError:
        logging.error("Could not import the Contract model from convert_cuad_dataset.")
        logging.error(
            "Make sure convert_cuad_dataset.py is in the same directory or adjust import path."
        )
        return

    title_to_id_map: Dict[str, int] = {}

    logging.info("Preparing data for insertion...")
    conn.execute("BEGIN TRANSACTION;")
    try:
        # --- Insert Contracts ---
        for contract_dict in tqdm(processed_contracts, desc="Processing Contracts"):
            # Re-create Pydantic object to use helpers
            try:
                contract_obj = Contract.model_validate(contract_dict)
            except ValidationError as e:
                logging.warning(
                    f"Skipping contract due to validation error: {contract_dict.get('title', 'N/A')}. Error: {e}"
                )
                continue

            # Insert contract
            cursor.execute(
                """
                INSERT INTO contracts (title, full_text)
                VALUES (?, ?)
                """,
                (contract_obj.title, contract_obj.full_text),
            )
            contract_pk_id = cursor.lastrowid
            title_to_id_map[contract_obj.title] = contract_pk_id  # Store mapping
            contract_count += 1

            # --- Prepare Annotations for this contract ---
            annotations_data = contract_obj.get_annotations_with_full_text_offsets()
            for ann in annotations_data:
                # Map title back to contract_id
                mapped_contract_id = title_to_id_map.get(ann["contract_title"])
                if mapped_contract_id is None:
                    logging.warning(
                        f"Could not find contract_id for title {ann['contract_title']} while processing annotation. Skipping annotation."
                    )
                    continue

                annotations_to_insert.append(
                    (
                        mapped_contract_id,
                        ann["clause_type"],
                        ann["text"],
                        ann["start_char_in_full_text"],
                        ann["end_char_in_full_text"],
                    )
                )
                annotation_count += 1

        # --- Bulk Insert Annotations ---
        logging.info(f"Bulk inserting {len(annotations_to_insert)} annotations...")
        if annotations_to_insert:
            cursor.executemany(
                """
                INSERT INTO annotations (contract_id, clause_type, text, start_char, end_char)
                VALUES (?, ?, ?, ?, ?)
                """,
                annotations_to_insert,
            )

        conn.commit()
        logging.info(
            f"Successfully inserted {contract_count} contracts and {annotation_count} annotations."
        )

    except sqlite3.Error as e:
        logging.error(f"Database population failed during transaction: {e}")
        conn.rollback()
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during population: {e}")
        conn.rollback()  # Rollback on any error during processing
        raise
    finally:
        conn.close()


def create_indexes_and_triggers(db_path: str):
    """Creates indexes and triggers AFTER data is populated."""
    logging.info(f"Creating indexes and FTS triggers for database: {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.executescript(SQL_CREATE_INDEXES_TRIGGERS)
        # Now populate the FTS table from the existing data
        logging.info("Populating FTS table...")
        cursor.execute(
            "INSERT INTO contracts_fts (rowid, title, full_text) SELECT id, title, full_text FROM contracts;"
        )
        conn.commit()
        logging.info("Indexes, triggers, and FTS table created successfully.")
    except sqlite3.Error as e:
        logging.error(f"Index/trigger creation failed: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def generate_database(
    processed_json_path: str = DEFAULT_PROCESSED_JSON_PATH,
    db_path: str = DEFAULT_DB_PATH,
    overwrite: bool = False,
):
    """
    Generates the SQLite database from the processed CUAD JSON file.

    Args:
        processed_json_path: Path to the 'processed_cuad_contracts.json'.
        db_path: Path where the SQLite database file should be created.
        overwrite: If True, any existing database file at db_path will be removed.
    """
    logging.info(
        f"Starting database generation from '{processed_json_path}' to '{db_path}'"
    )
    logging.info(f"Overwrite existing database: {overwrite}")

    # Handle existing DB file
    if overwrite and os.path.exists(db_path):
        logging.warning(f"Removing existing database file: {db_path}")
        try:
            os.remove(db_path)
        except OSError as e:
            logging.error(f"Error removing existing database file: {e}")
            return  # Exit if we can't remove the old DB when overwrite=True
    elif not overwrite and os.path.exists(db_path):
        logging.warning(
            f"Database file {db_path} exists and overwrite is False. Assuming file is up-to-date. Skipping generation."
        )
        return

    # 1. Load processed data from JSON
    processed_data = load_processed_data(processed_json_path)
    if not processed_data:
        logging.error("No processed data loaded. Cannot generate database.")
        return

    # 2. Create database schema (Tables only)
    create_database(db_path)

    # 3. Populate database with contracts and annotations
    populate_database(db_path, processed_data)

    # 4. Create Indexes and Triggers (including FTS population)
    create_indexes_and_triggers(db_path)

    logging.info(f"Database generation process completed for {db_path}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate CUAD SQLite Database")
    parser.add_argument(
        "--input-json",
        type=str,
        default=DEFAULT_PROCESSED_JSON_PATH,
        help=f"Path to the processed JSON file (default: {DEFAULT_PROCESSED_JSON_PATH})",
    )
    parser.add_argument(
        "--output-db",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Path for the output SQLite DB file (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the database file if it already exists.",
    )
    args = parser.parse_args()

    generate_database(
        processed_json_path=args.input_json,
        db_path=args.output_db,
        overwrite=args.overwrite,
    )
