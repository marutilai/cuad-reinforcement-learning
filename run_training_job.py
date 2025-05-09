# run_training_job.py (Adapted for CUAD)

import argparse
import sys
from pathlib import Path
from typing import Dict

import sky  # Ensure skypilot is installed: pip install skypilot

# Usage Example (from project root):
# uv run run_training_job.py 001 --accelerator A10G:1 --env-file ./cuad_qa/.env --fast


def load_env_file(env_path: str) -> Dict[str, str]:
    """Load a simple dotenv style file (KEY=VALUE per line)."""
    envs: Dict[str, str] = {}
    path = Path(env_path)
    if not path.exists():
        print(f"Warning: env file {env_path} does not exist – continuing without it.")
        return envs

    # Load environment variables relative to the env file's location
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                print(f"Warning: Skipping malformed line in {env_path}: {line}")
                continue
            key, val = line.split("=", 1)
            # Strip potential quotes from value
            val = val.strip().strip("'").strip('"')
            envs[key.strip()] = val
    print(f"Loaded environment variables from {env_path}")
    return envs


def main():
    parser = argparse.ArgumentParser(
        description="Launch a SkyPilot training job for the CUAD QA agent.",  # <<< Updated description
    )
    parser.add_argument(
        "run_id",
        help="The identifier for this training run (e.g., '001', 'baseline').",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use the --fast flag for sky.launch (skip provisioning if cluster is already up).",
    )
    parser.add_argument(
        "--env-file",
        # Default path assumes running from project root, .env inside cuad_qa
        default="cuad_qa/.env",
        help="Path to the environment file (default: cuad_qa/.env).",
    )
    parser.add_argument(
        "--idle-minutes",
        type=int,
        default=30,  # Reduced default idle time
        help="Idle minutes before autostop (default: 30).",
    )
    parser.add_argument(
        "--accelerator",
        default="A10G:1",  # Changed default to a more common/cheaper GPU
        help=(
            "Accelerator spec: '<TYPE>:<COUNT>'. Examples: 'A10G:1', 'A100:1', 'H100:1'. Default: 'A10G:1'."
        ),
    )
    parser.add_argument(
        "--cloud",
        default="runpod",  # Make cloud provider configurable
        choices=["runpod", "aws", "gcp", "lambda", "azure"],  # Add others as needed
        help="Cloud provider to use with SkyPilot (default: runpod).",
    )

    args = parser.parse_args()

    # --- Construct Cluster Name and Format RUN_ID ---
    # Allow non-integer run_ids, just use the string directly
    formatted_run_id = args.run_id
    # <<< Updated cluster name >>>
    cluster_name = f"cuad-qa-agent-{formatted_run_id}"

    # --- Define Task ---
    # This assumes run_training_job.py is in the ROOT of the cloned ART repo,
    # and your code is in ./cuad_qa/
    # Adjust workdir and file mounts if your structure differs.

    # --- Define Setup Script ---
    # Ensures uv, git, awscli are installed, clones ART if not mounted, installs project deps.
    setup_script = """
        echo "Setting up environment..."
        # Install uv if not present
        command -v uv >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.bashrc)

        # Check if ART repo is mounted/available, otherwise clone it
        if [ ! -d "ART" ]; then
            echo "Cloning ART repository..."
            git clone https://github.com/OpenPipe/ART.git
            cd ART # Change directory if cloning here
        else
            echo "ART directory found."
        fi

        # Install/Update dependencies within the workdir (e.g., ART/ or ART/examples/cuad_qa/)
        echo "Installing dependencies..."
        # Install ART library itself editable
        uv pip install --editable ./
        # Install dependencies from the project's pyproject.toml if it exists
        if [ -f "cuad_qa/pyproject.toml" ]; then
             echo "Installing dependencies from cuad_qa/pyproject.toml..."
             uv pip install -r cuad_qa/requirements.txt # Or sync pyproject.toml if you have one
        elif [ -f "pyproject.toml" ]; then
            # Fallback to root pyproject.toml if needed
            uv sync --all-extras
        else
             echo "Warning: No pyproject.toml found for project dependencies."
        fi
        uv pip install awscli # Ensure awscli for S3 backups

        # Run data preparation steps (ensure DB and Scenarios are ready)
        # These assume the necessary source files (CUADv1.json) are present or copied.
        # We might need to copy them using sky file_mounts or download them here.
        # For simplicity, assume they are copied via file_mounts for now.
        echo "Preparing data..."
        if [ -f "cuad_qa/raw_data/CUADv1.json" ]; then
            # Generate processed JSON
            python -m cuad_qa.data.convert_cuad_dataset --data-file cuad_qa/raw_data/CUADv1.json
            # Generate SQLite DB (using the generated processed JSON)
            python -m cuad_qa.data.local_contract_db --input-json cuad_qa/data/processed_CUADv1.json --overwrite
            # Generate and push scenarios (requires HF token in env)
            # Ensure HF_REPO_ID is set correctly for your user
            # Consider making HF_REPO_ID an env var passed from .env
            HF_REPO_ID="marutiagarwal/cuad-qa-scenarios" # Replace with your actual repo ID or load from ENV
            python -m cuad_qa.data.generate_cuad_scenarios --hf-repo-id ${HF_REPO_ID}
        else
            echo "WARNING: Raw data file cuad_qa/raw_data/CUADv1.json not found. Skipping data preparation."
            echo "Ensure data is prepared manually or mounted correctly."
        fi

        echo "Setup complete."
        """

    # --- Define Run Script ---
    run_script = f"""
        echo "Starting training run {formatted_run_id}..."
        # Set RUN_ID env var if train.py uses it (optional)
        export RUN_ID="{formatted_run_id}"
        # Execute the training script using python -m for proper module resolution
        python -m cuad_qa.train
        echo "Training run {formatted_run_id} finished."
        """

    # --- Define Base Environment Variables ---
    # These will be merged with/overridden by variables from the .env file
    base_envs = {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Faster HF downloads/uploads
        "WANDB_API_KEY": "",  # Optional: For Weights & Biases logging
        "OPENAI_API_KEY": "",  # Needed for the judge LLM
        "GOOGLE_API_KEY": "",  # Needed if using Gemini judge
        "OPENPIPE_API_KEY": "",  # Optional: For OpenPipe logging
        "RUN_ID": formatted_run_id,  # Pass run_id if needed by train.py
        "AWS_ACCESS_KEY_ID": "",  # For S3 backup
        "AWS_SECRET_ACCESS_KEY": "",  # For S3 backup
        "AWS_REGION": "",  # Optional: Specify S3 region
        "BACKUP_BUCKET": "",  # Required if using S3 backup
        "HF_TOKEN": "",  # Optional: Needed if pushing private HF datasets/models
        # Add any other ENV VARS your scripts might need
    }

    # --- Create SkyPilot Task ---
    task = sky.Task(
        name=f"cuad-qa-train-{formatted_run_id}",  # Give task a name
        # Workdir should be the root where cuad_qa dir exists
        # If running from ART root, workdir='.' might be correct.
        # If running from examples/, workdir='../..' might be needed. Adjust as necessary.
        workdir=".",  # <<< ADJUST if needed based on where you run `uv run run_training_job.py`
        setup=setup_script,
        run=run_script,
        envs=base_envs,  # Base envs; loaded .env file will update these
    )

    # --- Configure Resources ---
    try:
        acc_dict = sky.accelerators.get_accelerator_from_dot_string(args.accelerator)
        if acc_dict is None:
            raise ValueError(f"Invalid accelerator string: {args.accelerator}")
    except Exception as e:
        print(f"Error parsing accelerator: {e}", file=sys.stderr)
        sys.exit(1)

    task.set_resources(
        sky.Resources(
            cloud=sky.clouds.CLOUD_REGISTRY.from_str(args.cloud),  # Use selected cloud
            accelerators=acc_dict,
            # Optionally add memory/cpu requests if needed
            # memory="64G+",
        )
    )

    # --- Configure File Mounts ---
    # Mount the entire cuad_qa directory to the workdir on the remote instance
    # This assumes run_training_job.py is in the parent directory of cuad_qa
    # Adjust source path if needed.
    task.set_file_mounts(
        {
            # Mount your project code to the root of the workdir on the remote machine
            # Adjust '/cuad_qa' path if your local structure differs
            "/cuad_qa": "./cuad_qa",
            # Mount raw data if needed by setup script (ensure path exists locally)
            # '/cuad_qa/raw_data': './cuad_qa/raw_data', # Mount the raw data dir
        }
    )
    # Note: ART library is installed via setup script now, not mounted

    # --- Load and Update Environment Variables ---
    print(f"Loading environment variables from: {args.env_file}")
    loaded_envs = load_env_file(args.env_file)
    # Important: Update the task's envs *after* setting defaults
    task.update_envs(loaded_envs)
    # Ensure RUN_ID is correctly set (might be overridden by .env file, depends on desired priority)
    task.update_envs({"RUN_ID": formatted_run_id})

    print(f"Final task environment variables (excluding potentially sensitive):")
    for k, v in task.envs.items():
        if "KEY" not in k and "SECRET" not in k and "TOKEN" not in k:
            print(f"  {k}={v}")

    # --- Cancel Existing Jobs (Optional but Recommended) ---
    try:
        # Get status requires cluster name. If cluster doesn't exist, it raises an error.
        status = sky.status(cluster_name, refresh=False)  # Check if cluster exists
        if status:
            print(
                f"Cluster '{cluster_name}' already exists. Checking for running jobs..."
            )
            # Attempt to cancel any running/pending jobs on the cluster
            # Note: sky.queue might also error if cluster down; sky.cancel handles this better
            sky.cancel(cluster_name, all=True)
            print(f"Cancelled any existing jobs on '{cluster_name}'.")
    except Exception:
        # Catch errors if cluster doesn't exist (sky.status raises ClusterNotFoundError)
        # or if sky.cancel fails (e.g., cluster already down)
        print(f"Cluster '{cluster_name}' not found or no jobs to cancel.")
        pass  # Continue - cluster will be created by launch

    # --- Launch the Task ---
    print(f"\nLaunching training job on cluster '{cluster_name}'...")
    print(f"  Cloud: {args.cloud}")
    print(f"  Accelerator: {args.accelerator}")
    print(f"  Run ID: {formatted_run_id}")
    print(f"  Fast Launch: {args.fast}")
    print(f"  Auto-stop (idle): {args.idle_minutes} minutes")

    try:
        # Launch the job. `down=True` ensures cluster terminates after job or idle timeout.
        sky.launch(
            task,
            cluster_name=cluster_name,
            retry_until_up=True,  # Wait for cluster to be ready
            idle_minutes_to_autostop=args.idle_minutes,
            down=True,  # Terminate cluster when job finishes or idles
            stream_logs=True,  # Stream logs to console
            fast=args.fast,  # Reuse cluster if possible/requested
        )
        # Note: sky.launch with stream_logs=True blocks until completion.
        # If you need the request_id for detached execution, set stream_logs=False
        # and use sky.queue(request_id) / sky.logs(request_id) / sky.cancel(request_id) later.

        print(
            f"\nJob '{formatted_run_id}' on cluster '{cluster_name}' completed successfully."
        )

    except Exception as e:
        print(f"\nJob launch or execution failed: {e}", file=sys.stderr)
        # Optionally try to bring down the cluster if launch failed mid-way
        # try:
        #    print(f"Attempting to terminate cluster '{cluster_name}' due to failure...")
        #    sky.down(cluster_name)
        # except Exception as down_e:
        #    print(f"Failed to terminate cluster '{cluster_name}': {down_e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
