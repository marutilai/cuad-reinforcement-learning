# run_training_job.py (Adapted for CUAD)

import argparse
import sys
from pathlib import Path
from typing import Dict

import sky  # Ensure skypilot is installed: pip install skypilot

print(f"SkyPilot Version: {sky.__version__}")
print(f"SkyPilot Path: {sky.__file__}")

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
        echo "SKYPILOT SETUP SCRIPT STARTED"
        set -e # Exit immediately if a command exits with a non-zero status.

        echo "INFO: Initial PATH: $PATH"
        echo "INFO: Initial PWD: $(pwd)" # Will be ~/sky_workdir or similar
        cd project_code # <<< NAVIGATE INTO YOUR MOUNTED PROJECT
        echo "INFO: Changed PWD to: $(pwd)" # Will be ~/sky_workdir/project_code
        echo "INFO: Initial python: $(which python || echo 'python not found')"
        echo "INFO: Initial pip: $(which pip || echo 'pip not found')"

        # Install Poetry
        if ! command -v poetry &> /dev/null; then
            echo "INFO: Poetry not found, installing..."
            curl -sSL https://install.python-poetry.org | python3 -
            # Attempt to add poetry to PATH immediately for this script session
            # This is often $HOME/.local/bin or similar.
            # The exact path can vary by Linux distribution / base image.
            if [ -d "$HOME/.local/bin" ]; then
                export PATH="$HOME/.local/bin:$PATH"
            elif [ -d "$HOME/bin" ]; then # Another common location
                export PATH="$HOME/bin:$PATH"
            fi
            # Also try to ensure it's in .bashrc for subsequent logins to the VM (if kept up)
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc # Or ~/.profile, ~/.zshrc
        else
            echo "INFO: Poetry already installed."
        fi
        echo "INFO: Poetry version: $(poetry --version || echo 'poetry --version failed')"
        echo "INFO: Path to poetry: $(which poetry || echo 'poetry not found in PATH after potential install')"


        # Ensure we are in the project root (where pyproject.toml for cuad_qa is)
        # SkyPilot's file_mounts {'/': '.'} copies the local project root to '/' on remote.
        # So, pyproject.toml for cuad_qa is at /pyproject.toml on remote.
        echo "INFO: Listing root directory: $(ls -la /)"
        if [ ! -f "./pyproject.toml" ]; then
            echo "ERROR: ./pyproject.toml not found! Check SkyPilot file_mounts and workdir."
            exit 1
        fi

        echo "INFO: Installing project dependencies using Poetry from ./pyproject.toml..."
        # Use -vvv for very verbose output if still having issues
        poetry install --no-dev --all-extras --no-root
        echo "INFO: Poetry install finished."

        # Verify key package versions *within the Poetry environment*
        echo "INFO: Verifying package versions with 'poetry run pip show':"
        poetry run pip show numpy torch openpipe-art skypilot
        # Specifically check NumPy version managed by Poetry
        NUMPY_VERSION_POETRY=$(poetry run pip show numpy | grep Version | awk '{print $2}')
        echo "INFO: NumPy version according to 'poetry run pip show': $NUMPY_VERSION_POETRY"

        # Force uninstall any other numpy versions and reinstall the one Poetry wants
        # This is an aggressive step to try and resolve conflicts.
        echo "INFO: Attempting to ensure correct NumPy version..."
        poetry run pip uninstall -y numpy # Uninstall numpy from poetry env (if any)
        # poetry run pip uninstall -y $(pip list | grep -i numpy | awk '{print $1}') # Try to remove any system/other numpy
        # sudo apt-get remove -y python3-numpy # If it's a system package (less likely on cloud VMs)
        poetry run pip install "numpy~=1.26.4" # Force install your pinned version
        echo "INFO: NumPy version after explicit reinstall: $(poetry run pip show numpy | grep Version)"


        # Data Preparation (run within Poetry's environment)
        echo "INFO: Preparing data..."
        RAW_DATA_FILE="cuad_qa/raw_data/CUADv1.json" # Path is now relative to 'project_code'
        PROCESSED_JSON_FILE="cuad_qa/data/processed_CUADv1.json"
        DB_FILE="cuad_qa/data/cuad_contracts.db"

        if [ -f "$RAW_DATA_FILE" ]; then
            echo "INFO: Found raw data file: $RAW_DATA_FILE"
            poetry run python -m cuad_qa.data.convert_cuad_dataset --data-file "$RAW_DATA_FILE"
            if [ -f "$PROCESSED_JSON_FILE" ]; then
                poetry run python -m cuad_qa.data.local_contract_db --input-json "$PROCESSED_JSON_FILE" --overwrite
                if [ -f "$DB_FILE" ]; then
                    HF_REPO_ID_VAR="${HF_REPO_ID:-marutiagarwal/cuad-qa-scenarios}"
                    poetry run python -m cuad_qa.data.generate_cuad_scenarios --hf-repo-id "${HF_REPO_ID_VAR}"
                else echo "WARNING: DB file $DB_FILE not found. Skipping scenario generation."; fi
            else echo "WARNING: Processed JSON $PROCESSED_JSON_FILE not found. Skipping DB/scenario gen."; fi
        else echo "WARNING: Raw data file $RAW_DATA_FILE not found. Skipping data prep."; fi

        echo "SKYPILOT SETUP SCRIPT FINISHED"
    """

    # Ensure your run_script uses poetry run:
    run_script = f"""
        echo "SKYPILOT RUN SCRIPT STARTED"
        set -e
        export PATH="$HOME/.local/bin:$PATH" # Ensure poetry is in PATH for run script too

        cd project_code # <<< NAVIGATE INTO YOUR MOUNTED PROJECT

        # Verify environment for the run script
        echo "INFO (run_script): Current directory: $(pwd)"
        echo "INFO (run_script): Python being used: $(poetry run which python)"
        echo "INFO (run_script): NumPy version: $(poetry run pip show numpy | grep Version)"

        echo "INFO (run_script): Starting training run {formatted_run_id}..."
        export RUN_ID="{formatted_run_id}"
        poetry run python -m cuad_qa.train # Key command
        echo "INFO (run_script): Training run {formatted_run_id} finished."
        echo "SKYPILOT RUN SCRIPT FINISHED"
    """

    # --- Define Base Environment Variables ---
    # These will be merged with/overridden by variables from the .env file
    base_envs = {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Faster HF downloads/uploads
        "WANDB_API_KEY": "",  # Optional: For Weights & Biases logging
        "OPENAI_API_KEY": "",  # Needed for the judge LLM
        # "GOOGLE_API_KEY": "",  # Needed if using Gemini judge
        "OPENPIPE_API_KEY": "",  # Optional: For OpenPipe logging
        "RUN_ID": formatted_run_id,  # Pass run_id if needed by train.py
        # "AWS_ACCESS_KEY_ID": "",  # For S3 backup
        # "AWS_SECRET_ACCESS_KEY": "",  # For S3 backup
        # "AWS_REGION": "",  # Optional: Specify S3 region
        # "BACKUP_BUCKET": "",  # Required if using S3 backup
        "HUGGINGFACE_HUB_TOKEN": "",  # Optional: Needed if pushing private HF datasets/models
        "HF_REPO_ID": "marutiagarwal/cuad-qa-scenarios",  # Make HF_REPO_ID configurable via .env for the setup script
        "NP_BOOLEAN_WARNING_AND_FUTURE_ERROR_ALLOWED": "1",  # Important for some libraries like transformers/hf with numpy 2.0 issues
        # Add any other ENV VARS your scripts might need
    }

    # --- Create SkyPilot Task ---
    task = sky.Task(
        name=f"cuad-qa-train-{formatted_run_id}",  # Give task a name
        # Workdir should be the root where cuad_qa dir exists
        # If running from ART root, workdir='.' might be correct.
        # If running from examples/, workdir='../..' might be needed. Adjust as necessary.
        # workdir="/",  # Run commands from the root of the mounted project
        setup=setup_script,
        run=run_script,
        envs=base_envs,  # Base envs; loaded .env file will update these
    )

    # --- Configure Resources ---
    def _parse_accelerator(spec: str) -> Dict[str, int]:
        """Parse an accelerator spec of the form 'TYPE:COUNT' (COUNT optional)."""
        if ":" in spec:
            name, count_str = spec.split(":", 1)
            try:
                count = int(count_str)
                if count <= 0:
                    raise ValueError("Count must be positive.")
            except ValueError as exc:
                raise ValueError(
                    f"Invalid accelerator count in spec '{spec}'. Must be a positive int."
                ) from exc
        else:
            name, count = spec, 1  # Default to 1 if no count specified
        return {name: count}

    try:
        acc_dict = _parse_accelerator(args.accelerator)
        if acc_dict is None:
            raise ValueError(f"Invalid accelerator string: {args.accelerator}")
    except Exception as e:
        print(f"Error parsing accelerator: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Select Cloud Object ---
    cloud_instance = None
    if args.cloud.lower() == "aws":
        cloud_instance = sky.clouds.AWS()
    elif args.cloud.lower() == "gcp":
        cloud_instance = sky.clouds.GCP()
    elif args.cloud.lower() == "azure":
        cloud_instance = sky.clouds.Azure()
    elif args.cloud.lower() == "runpod":
        cloud_instance = sky.clouds.RunPod()
    elif args.cloud.lower() == "lambda":
        cloud_instance = sky.clouds.Lambda()
    # Add other clouds as needed from sky.clouds module
    else:
        print(
            f"Error: Unsupported or unknown cloud provider specified: {args.cloud}",
            file=sys.stderr,
        )
        print(f"Supported examples: aws, gcp, azure, runpod, lambda", file=sys.stderr)
        sys.exit(1)

    task.set_resources(
        sky.Resources(
            cloud=cloud_instance,  # Use selected cloud
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
            "project_code": "."
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

    # Ensure HF_REPO_ID from .env is passed if set, otherwise use default from base_envs
    if "HF_REPO_ID" in loaded_envs:
        task.update_envs({"HF_REPO_ID": loaded_envs["HF_REPO_ID"]})

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
        request_id = sky.launch(
            task,
            cluster_name=cluster_name,
            retry_until_up=True,  # Wait for cluster to be ready
            idle_minutes_to_autostop=args.idle_minutes,
            down=True,  # Terminate cluster when job finishes or idles
            fast=args.fast,  # Reuse cluster if possible/requested
        )
        print(f"SkyPilot launch request submitted. Request ID: {request_id}")

        # Note: sky.launch with stream_logs=True blocks until completion.
        # If you need the request_id for detached execution, set stream_logs=False
        # and use sky.queue(request_id) / sky.logs(request_id) / sky.cancel(request_id) later.
        print(f"Streaming logs for request ID: {request_id}...")
        sky.stream_logs(request_id=request_id, follow=True)

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
