# cuad_qa/contract_search_tools.py

import os
import sqlite3
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from cuad_qa.data.local_contract_db import DEFAULT_DB_PATH

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Global connection (consider connection pooling for higher concurrency if needed)
conn = None


def get_conn():
    """Gets a thread-safe, read-only connection to the contract database."""
    global conn
    if conn is None:
        if not os.path.exists(DEFAULT_DB_PATH):
            logging.error(f"Database file not found at: {DEFAULT_DB_PATH}")
            raise FileNotFoundError(f"Database file not found: {DEFAULT_DB_PATH}")
        logging.info(f"Establishing read-only connection to: {DEFAULT_DB_PATH}")
        # Connect in read-only mode ('ro') for safety during rollouts
        conn = sqlite3.connect(
            f"file:{DEFAULT_DB_PATH}?mode=ro", uri=True, check_same_thread=False
        )
    return conn


@dataclass
class ContractSearchResult:
    """Dataclass to hold results from a contract search."""

    contract_id: int  # The primary key from the 'contracts' table
    title: str
    snippet: str


# --- Tool Functions ---


def search_contracts(
    keywords: List[str],
    max_results: int = 10,
) -> List[ContractSearchResult]:
    """
    Searches the contract database based on keywords in the title or full text.

    Args:
        keywords: A list of keywords that must all appear in the title or body.
        max_results: The maximum number of results to return. Cannot exceed 10.

    Returns:
        A list of ContractSearchResult objects, each containing 'contract_id', 'title', and 'snippet'.
        Returns an empty list if no results are found or an error occurs.
    """
    if not keywords:
        logging.warning("No keywords provided for search_contracts.")
        return []  # Return empty list instead of raising error to be more robust for agent

    if not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords):
        logging.error(f"Keywords must be a list of strings, got: {keywords}")
        raise TypeError("Keywords must be a list of strings.")

    if max_results > 10:
        logging.warning("max_results exceeds 10, capping at 10.")
        max_results = 10
    if max_results <= 0:
        logging.warning("max_results must be positive, setting to default 10.")
        max_results = 10

    cursor = get_conn().cursor()

    # --- Build FTS Query ---
    # FTS5 default is AND. Escape quotes for safety.
    # Search across both title and full_text (column index -1 in snippet)
    try:
        fts_query = " ".join(f'''"{k.replace('"', '""')}"''' for k in keywords)
        params: List[Any] = [fts_query, max_results]

        # snippet(<table>, <column_index>, <highlight_start>, <highlight_end>, <ellipsis>, <tokens>)
        # column_index = -1 searches all indexed columns (title, full_text)
        # tokens = 15 gives about 15 words around the match
        sql = """
            SELECT
                c.id,       -- Contract primary key
                c.title,    -- Contract title
                snippet(contracts_fts, -1, '<b>', '</b>', ' ... ', 15) as snippet
            FROM
                contracts c JOIN contracts_fts fts ON c.id = fts.rowid
            WHERE
                fts.contracts_fts MATCH ?
            ORDER BY
                rank -- Default FTS5 ordering by relevance
            LIMIT ?;
        """

        logging.debug(f"Executing SQL: {sql}")
        logging.debug(f"With params: {params}")
        cursor.execute(sql, params)
        results = cursor.fetchall()

        # Format results
        formatted_results = [
            ContractSearchResult(contract_id=row[0], title=row[1], snippet=row[2])
            for row in results
        ]
        logging.info(
            f"Contract search found {len(formatted_results)} results for keywords: {keywords}."
        )
        return formatted_results

    except sqlite3.Error as e:
        logging.error(f"Database error during contract search: {e}")
        return []  # Return empty list on DB error
    except Exception as e:
        logging.error(f"Unexpected error during contract search: {e}")
        return []  # Return empty list on other errors


def read_contract(contract_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieves a single contract by its primary key ID from the database.

    Args:
        contract_id: The unique primary key ID of the contract to retrieve.

    Returns:
        A dictionary containing 'id', 'title', and 'full_text' of the found contract,
        or None if the contract is not found or an error occurs.
    """
    if not isinstance(contract_id, int) or contract_id <= 0:
        logging.error(
            f"Invalid contract_id provided to read_contract: {contract_id}. Must be a positive integer."
        )
        return None  # Return None for invalid input

    cursor = get_conn().cursor()

    # --- Query for Contract Details ---
    contract_sql = """
        SELECT id, title, full_text
        FROM contracts
        WHERE id = ?;
    """
    try:
        cursor.execute(contract_sql, (contract_id,))
        contract_row = cursor.fetchone()

        if not contract_row:
            logging.warning(f"Contract with id '{contract_id}' not found.")
            return None

        # Unpack row and return as dictionary
        (id_val, title, full_text) = contract_row
        contract_data = {"id": id_val, "title": title, "full_text": full_text}
        logging.info(
            f"Successfully read contract ID: {contract_id}, Title: {title[:50]}..."
        )
        return contract_data

    except sqlite3.Error as e:
        logging.error(f"Database error reading contract id {contract_id}: {e}")
        return None  # Return None on DB error
    except Exception as e:
        logging.error(f"Unexpected error reading contract id {contract_id}: {e}")
        return None  # Return None on other errors


# Example Usage (for testing the script directly)
if __name__ == "__main__":
    print("--- Testing Contract Search Tools ---")

    # Ensure DB exists before testing
    if not os.path.exists(DEFAULT_DB_PATH):
        print(f"ERROR: Database file not found at {DEFAULT_DB_PATH}")
        print("Please run local_contract_db.py first to create the database.")
    else:
        # Test Search
        print("\nTesting search_contracts...")
        search_keywords = ["governing law"]  # Example keywords
        search_results = search_contracts(keywords=search_keywords, max_results=3)
        if search_results:
            print(f"Found {len(search_results)} results for '{search_keywords}':")
            for i, result in enumerate(search_results):
                print(
                    f"  {i+1}. ID: {result.contract_id}, Title: {result.title}, Snippet: {result.snippet}"
                )
        else:
            print(f"No results found for '{search_keywords}'.")

        # Test Read (using an ID from the search results if available)
        print("\nTesting read_contract...")
        if search_results:
            test_id = search_results[0].contract_id
            print(f"Attempting to read contract with ID: {test_id}")
            contract_data = read_contract(contract_id=test_id)
            if contract_data:
                print("Contract found:")
                print(f"  ID: {contract_data['id']}")
                print(f"  Title: {contract_data['title']}")
                print(
                    f"  Full Text (first 200 chars): {contract_data['full_text'][:200]}..."
                )
            else:
                print(f"Contract with ID {test_id} not found or error occurred.")
        else:
            print(
                "Skipping read test as no contracts were found in the previous search."
            )

        # Test Read with invalid ID
        print("\nTesting read_contract with invalid ID...")
        invalid_id = -1
        contract_data = read_contract(contract_id=invalid_id)
        if contract_data is None:
            print(f"Correctly handled invalid ID {invalid_id} (returned None).")
        else:
            print(f"ERROR: Did not correctly handle invalid ID {invalid_id}.")

    print("\n--- Test complete ---")
