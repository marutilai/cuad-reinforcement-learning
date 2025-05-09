# cuad_qa/evaluate/benchmark.py

import art
import polars as pl
import logging
from typing import List

# Import from relative paths within the cuad_qa package
# These imports assume benchmark.py is inside an 'evaluate' subdirectory
from cuad_qa.rollout import rollout
from cuad_qa.data.query_iterators import load_clause_finding_scenarios
from cuad_qa.data.generate_cuad_scenarios import ClauseFindingScenario

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


async def benchmark_cuad_model(  # Renamed function for clarity
    model: art.Model,
    limit: int = 100,
    swallow_exceptions: bool = True,
    split: str = "test",  # Add parameter to choose split, default to test
) -> pl.DataFrame:
    """
    Benchmarks a model on the CUAD clause finding task using a specified dataset split.

    Args:
        model: The art.Model instance to benchmark (can be trainable or just for inference).
        limit: The maximum number of scenarios to evaluate.
        swallow_exceptions: If True, ignores errors during individual rollouts and continues.
                            If False, raises the first exception encountered.
        split: The dataset split to use ('test' or 'train').

    Returns:
        A Polars DataFrame containing the average metrics across the evaluated trajectories.
        Returns an empty DataFrame if loading scenarios fails or no valid trajectories are produced.
    """
    logging.info(
        f"Starting benchmark for model '{model.name}' on '{split}' split (limit: {limit})."
    )

    # 1. Load the appropriate scenarios for CUAD Clause Finding
    try:
        # Use our specific scenario loader function
        scenarios: List[ClauseFindingScenario] = load_clause_finding_scenarios(
            split=split,
            limit=limit,
            shuffle=False,  # Typically don't shuffle evaluation sets
        )
    except Exception as e:
        logging.error(
            f"Failed to load scenarios for benchmarking (split='{split}'): {e}"
        )
        # Return an empty DataFrame or re-raise depending on desired behavior
        return pl.DataFrame()

    if not scenarios:
        logging.warning(
            f"No scenarios found for split '{split}'. Returning empty benchmark results."
        )
        return pl.DataFrame()

    logging.info(f"Benchmarking on {len(scenarios)} scenarios...")

    # 2. Gather trajectories by running the CUAD rollout function for each scenario
    eval_trajectories = await art.gather_trajectories(
        # Ensure we are calling OUR adapted rollout function
        (rollout(model, scenario) for scenario in scenarios),
        pbar_desc=f"Benchmarking {model.name} ({split} split)",
        max_exceptions=limit if swallow_exceptions else 0,  # Control error handling
    )

    # 3. Filter out any exceptions that might have occurred if swallow_exceptions=True
    valid_trajectories = [t for t in eval_trajectories if isinstance(t, art.Trajectory)]
    num_valid = len(valid_trajectories)
    num_errors = len(eval_trajectories) - num_valid

    if num_errors > 0:
        logging.warning(f"Benchmark encountered {num_errors} errors during rollouts.")
    if not valid_trajectories:
        logging.warning("No valid trajectories were generated during benchmarking.")
        # Return DataFrame indicating 0 trajectories evaluated
        return pl.DataFrame({"n_trajectories": [0]})

    logging.info(
        f"Benchmark completed. Successfully generated {num_valid} valid trajectories."
    )

    # 4. Log the individual valid trajectories (useful for later analysis)
    # Use model.log() - it should handle interaction with the registered API/backend correctly.
    # No need to check model._backend or model._api directly usually.
    try:
        await model.log(valid_trajectories)
        logging.info(
            f"Logged {num_valid} benchmark trajectories for model '{model.name}'."
        )
    except Exception as e:
        # Log error but continue with metric calculation
        logging.warning(
            f"Failed to log benchmark trajectories for model '{model.name}'. Error: {e}"
        )

    # 5. Extract metrics and rewards into a Polars DataFrame
    # Include 'reward' alongside metrics from the FinalRubric
    metrics_list = []
    for t in valid_trajectories:
        if hasattr(t, "metrics") and isinstance(t.metrics, dict):
            # Ensure reward exists before adding
            reward_val = t.reward if hasattr(t, "reward") else None
            metrics_list.append({**t.metrics, "reward": reward_val})
        elif hasattr(t, "reward"):
            # Handle cases where only reward might be present (less likely with our rubric)
            metrics_list.append({"reward": t.reward})
        # Else: Trajectory might be malformed, skip it for metrics calculation

    if not metrics_list:
        logging.warning("No metrics found in valid trajectories.")
        # Return with trajectory count, but no metric averages
        return pl.DataFrame({"n_trajectories": [num_valid]})

    metrics_df = pl.DataFrame(metrics_list)

    # 6. Calculate average metrics across all valid trajectories
    avg_metrics_exprs = []
    for col in metrics_df.columns:
        # Check if the column is numeric before attempting mean aggregation
        is_numeric = pl.datatypes.FLOAT_DTYPES.union(pl.datatypes.INTEGER_DTYPES)
        # Skip None types if they exist after the list comprehension
        if metrics_df[col].dtype != pl.Null and metrics_df[col].dtype in is_numeric:
            avg_metrics_exprs.append(pl.mean(col).alias(col))
        # else: log or handle non-numeric columns if needed

    if not avg_metrics_exprs:
        logging.warning("No numeric metrics found to average.")
        # Return with trajectory count
        return pl.DataFrame({"n_trajectories": [num_valid]})

    # Compute averages and add the count of valid trajectories evaluated
    avg_metrics_df = metrics_df.select(avg_metrics_exprs).with_columns(
        pl.lit(num_valid).alias("n_trajectories")
    )

    logging.info(f"Average benchmark metrics for '{model.name}':\n{avg_metrics_df}")
    return avg_metrics_df


# Example usage within train.py (or a separate evaluation script)
async def example_usage():
    from cuad_qa.project_configs import ProjectPolicyConfig

    # Assume 'cuad_agent_001' is your trained model object
    # Or define a non-trainable model for benchmarking (e.g., GPT-4o)
    benchmark_target_model = art.Model(
        name="gpt-4o-mini-benchmark",  # Give it a unique name for logging
        project="cuad_clause_agent",  # Match project name
        config=ProjectPolicyConfig(
            litellm_model_name="openai/gpt-4o-mini",  # Specify the model
            use_tools=True,  # Must match agent's tool usage
        ),
    )
    # Register with LocalAPI if needed, especially if benchmarking locally after training
    # api = art.LocalAPI()
    # await benchmark_target_model.register(api)

    print(f"\n--- Running Example Benchmark for {benchmark_target_model.name} ---")
    results_df = await benchmark_cuad_model(
        benchmark_target_model, limit=10, split="test"
    )  # Use test split
    print("\nBenchmark Results (DataFrame):")
    print(results_df)


if __name__ == "__main__":
    import asyncio
    # Ensure dotenv is loaded if testing standalone and need API keys
    # from dotenv import load_dotenv
    # load_dotenv()
    # Ensure DB exists
    # from cuad_qa.data.local_contract_db import generate_database
    # generate_database(overwrite=False)

    # Run the example usage
    # asyncio.run(example_usage())
    pass  # Keep __main__ empty or add specific test calls as needed
