# CUAD QA Processing Modeling

Contract Understanding Atticus Dataset (CUAD) for question answering tasks.

## Quick Start

```bash
# Install dependencies
poetry install

# Process CUAD dataset
poetry run python cuad-qa/data/convert_cuad_dataset.py --data-file cuad-qa/raw_data/CUADv1.json

# Process test set only (for quick testing)
poetry run python cuad-qa/data/convert_cuad_dataset.py --data-file cuad-qa/raw_data/test.json --max-contracts 5
```
