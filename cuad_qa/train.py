# cuad_qa/train.py (Final Version)

import art
import asyncio
import os
import logging
from dotenv import load_dotenv
from typing import List

# Corrected imports for CUAD project
from .rollout import rollout
from .data.query_iterators import load_clause_finding_scenarios
from .data.generate_cuad_scenarios import ClauseFindingScenario
from .data.local_contract_db import generate_database
from art.utils import iterate_dataset
from .project_configs import ProjectPolicyConfig, TrainingConfig
from .evaluate.benchmark import (
    benchmark_cuad_model,
)  # <<< IMPORT ACTUAL BENCHMARK FUNCTION
from art.local import LocalAPI  # <<< CORRECTED IMPORT

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Define the Model to Train ---
# Start with one configuration for our CUAD agent
cuad_agent_001 = art.TrainableModel(
    name="cuad-agent-001",
    project="cuad_clause_agent",
    base_model="Qwen/Qwen2.5-14B-Instruct",  # Consider changing if needed
    config=ProjectPolicyConfig(
        max_turns=10,
        log_to_openpipe=False,  # Enable if using OpenPipe
        use_tools=True,
        training_config=TrainingConfig(
            # Adjusted based on final ART·E config
            trajectories_per_group=4,
            groups_per_step=12,
            learning_rate=1.2e-5,
            eval_steps=30,  # Evaluate every 30 steps
            val_set_size=200,  # Number of validation scenarios
            training_dataset_size=10000,  # Max training scenarios to load
            num_epochs=3,  # Number of passes over training data
        ),
    ),
)

# cuad_agent_002_more_turns = cuad_agent_001.model_copy(deep=True)
# cuad_agent_002_more_turns.name = "cuad-agent-002-more-turns"
# assert isinstance(cuad_agent_002_more_turns.config, ProjectPolicyConfig)
# cuad_agent_002_more_turns.config.max_turns = 15

# cuad_agent_003_lower_lr = cuad_agent_001.model_copy(deep=True)
# cuad_agent_003_lower_lr.name = "cuad-agent-003-lower-lr"
# assert isinstance(cuad_agent_003_lower_lr.config, ProjectPolicyConfig)
# assert cuad_agent_003_lower_lr.config.training_config is not None
# cuad_agent_003_lower_lr.config.training_config.learning_rate = 5e-6


# --- Main Training Function ---
async def run_training(model: art.TrainableModel):
    """Orchestrates the ART training loop for the CUAD agent."""
    logging.info(f"Starting training run for model: {model.name}")

    # 1. Ensure Database Exists
    logging.info("Ensuring CUAD contract database exists...")
    try:
        # Call DB generation script (won't overwrite if exists and overwrite=False)
        generate_database(overwrite=False)
        logging.info("Database check/generation complete.")
    except Exception as e:
        logging.error(f"Failed to generate/verify database: {e}")
        return

    # 2. Setup ART API and Model Registration
    assert isinstance(
        model.config, ProjectPolicyConfig
    ), "Model config must be ProjectPolicyConfig"
    if model.config.training_config is None:
        raise ValueError("Training config is not set for a trainable model")

    api = LocalAPI()  # <<< CORRECTED API CLASS
    await model.register(api)
    logging.info(f"Model {model.name} registered with LocalAPI.")

    # 3. Load Checkpoints from S3 (Optional)
    # --- S3 PULL REMOVED ---
    # s3_bucket = os.environ.get("BACKUP_BUCKET")
    # if s3_bucket:
    #     logging.info(f"Attempting to pull previous checkpoints from S3 bucket: `{s3_bucket}`")
    #     try:
    #         await api._experimental_pull_from_s3(model, s3_bucket=s3_bucket, verbose=True)
    #         logging.info(f"Successfully pulled checkpoints for {model.name}.")
    #     except Exception as e:
    #         logging.warning(f"Could not pull checkpoints from S3. Starting fresh. Error: {e}")
    # else:
    #     logging.info("BACKUP_BUCKET not set. Skipping S3 checkpoint loading. Using local checkpoints if any.")
    logging.info("Using local checkpoints if any exist, or starting fresh.")

    # 4. Load Training & Validation Data (Scenarios)
    # Note: Validation scenarios are loaded within benchmark_cuad_model now
    logging.info("Loading training scenarios...")
    try:
        train_scenarios: List[ClauseFindingScenario] = load_clause_finding_scenarios(
            split="train",
            limit=model.config.training_config.training_dataset_size,
            shuffle=True,
        )
        if not train_scenarios:
            logging.error("No training scenarios loaded. Aborting training.")
            return
        logging.info(f"Loaded {len(train_scenarios)} training scenarios.")
    except Exception as e:
        logging.error(f"Failed to load training scenarios: {e}")
        return

    # 5. Create Data Iterator
    initial_step = await model.get_step()
    logging.info(f"Starting training from step: {initial_step}")
    train_iterator = iterate_dataset(
        dataset=train_scenarios,  # Pass loaded scenarios
        groups_per_step=model.config.training_config.groups_per_step,
        num_epochs=model.config.training_config.num_epochs,
        initial_step=initial_step,
    )

    # 6. Training Loop
    logging.info("Starting training loop...")
    for batch, epoch, global_step, epoch_step in train_iterator:
        logging.info(
            f"--- Step {global_step} (Epoch {epoch}, Batch {epoch_step+1}) ---"
        )

        # --- Periodic Evaluation & Checkpointing ---
        # Evaluate slightly *before* the step number for clarity (e.g., eval after step 29 runs before step 30 training)
        if global_step > initial_step and (
            global_step % model.config.training_config.eval_steps == 0
        ):
            logging.info(f"\n--- Evaluating model at Step {global_step} ---")
            # <<< CALL ACTUAL BENCHMARK FUNCTION >>>
            await benchmark_cuad_model(
                model,
                limit=model.config.training_config.val_set_size,
                split="test",  # Explicitly use test split for evaluation
            )

            # --- S3 PUSH REMOVED ---
            # if s3_bucket:
            #     logging.info(f"Pushing checkpoint and logs to S3 bucket: `{s3_bucket}`")
            #     try:
            #         await api._experimental_push_to_s3(model, s3_bucket=s3_bucket)
            #         logging.info("Successfully pushed to S3.")
            #     except Exception as e:
            #         logging.warning(f"Could not push checkpoint to S3. Error: {e}")
            # else:
            #     logging.info("Skipping S3 push (BACKUP_BUCKET not set). Model checkpoints saved locally.")
            logging.info("Model checkpoints and trajectories saved locally.")

        # --- Gather Trajectories for Training Step ---
        logging.info(f"Gathering trajectories for {len(batch)} scenarios...")
        # Use the batch directly from the iterator
        groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    (
                        rollout(model, scenario)
                        for _ in range(
                            model.config.training_config.trajectories_per_group
                        )
                    )
                )
                for scenario in batch  # Iterate through scenarios in the current batch
            ),
            pbar_desc=f"Rollouts Step {global_step}",
        )
        logging.info(f"Gathered {len(groups)} trajectory groups.")

        # --- Perform Training Step ---
        if not groups:
            logging.warning(
                f"No trajectory groups gathered for step {global_step}. Skipping training step."
            )
            continue

        logging.info("Performing training step...")
        await model.train(
            groups,
            config=art.TrainConfig(
                learning_rate=model.config.training_config.learning_rate
            ),
        )
        logging.info(f"Training step {global_step} complete.")

    # 7. Final Evaluation & Save
    logging.info("--- Training Loop Finished ---")
    logging.info("Running final evaluation...")
    # <<< CALL ACTUAL BENCHMARK FUNCTION >>>
    await benchmark_cuad_model(
        model, limit=model.config.training_config.val_set_size, split="test"
    )

    # --- S3 PUSH REMOVED ---
    # if s3_bucket:
    #     logging.info(f"Pushing final model checkpoint to S3 bucket: `{s3_bucket}`")
    #     try:
    #         await api._experimental_push_to_s3(model, s3_bucket=s3_bucket)
    #         logging.info("Successfully pushed final model to S3.")
    #     except Exception as e:
    #         logging.warning(f"Could not push final model checkpoint to S3. Error: {e}")
    # else:
    #     logging.info("Skipping final S3 push (BACKUP_BUCKET not set). Final model checkpoint saved locally.")
    logging.info("Final model checkpoint and trajectories saved locally.")


# --- Script Entry Point ---
if __name__ == "__main__":
    # No need to check for BACKUP_BUCKET if not using S3
    # if not os.environ.get("BACKUP_BUCKET"):
    #      logging.warning("BACKUP_BUCKET env var not set. S3 saving/loading disabled.")

    config_to_run = cuad_agent_001
    logging.info(f"Selected configuration to run: {config_to_run.name}")

    try:
        asyncio.run(run_training(config_to_run))
    except Exception as main_error:
        logging.error(f"Training process failed: {main_error}", exc_info=True)
        exit(1)
