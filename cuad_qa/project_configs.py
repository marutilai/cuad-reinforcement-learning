# cuad_qa/project_configs.py

from pydantic import BaseModel, Field
from typing import Optional


# --- Training Loop Configuration ---
class TrainingConfig(BaseModel):
    """Hyperparameters and settings for the ART training process."""

    trajectories_per_group: int = Field(
        default=6,
        description="Number of rollouts to perform for each scenario within a group (for GRPO).",
    )
    groups_per_step: int = Field(
        default=1,
        description="Number of scenario groups to process in a single training step.",
    )
    learning_rate: float = Field(
        default=1.2e-5, description="Learning rate for the LoRA adapter updates."
    )
    eval_steps: int = Field(
        default=30, description="Frequency (in training steps) to run evaluation."
    )
    val_set_size: int = Field(
        default=100, description="Number of scenarios to use for validation."
    )
    training_dataset_size: int = Field(
        default=4000,  # Should correspond roughly to the number of available train scenarios
        description="Maximum number of scenarios to load from the training dataset.",
    )
    num_epochs: int = Field(
        default=4,
        description="Number of times to iterate over the training dataset.",
    )


# --- Agent Policy & Generation Configuration ---
class ProjectPolicyConfig(BaseModel):
    """Configuration controlling the agent's behavior during rollouts and general project settings."""

    max_turns: int = Field(
        default=10,
        description="Maximum number of steps (LLM calls + tool executions) per rollout.",
    )
    max_tokens: int = Field(
        default=2048,
        description="Maximum number of tokens the LLM can generate in a single turn.",
    )
    log_to_openpipe: bool = Field(
        default=False, description="Flag to enable logging rollouts to OpenPipe."
    )
    litellm_model_name: Optional[str] = Field(
        default=None,
        description="Specify a LiteLLM model string (e.g., 'openai/gpt-4o') for benchmarking non-ART models or if base model differs.",
    )
    use_tools: bool = Field(
        default=True,
        description="Whether the agent uses OpenAI-style tool calling format (True) or expects raw JSON output (False).",
    )
    stupid_simple_reward_fn: bool = Field(
        default=False,
        description="Ablation flag: If True, use a very basic reward function (e.g., 1 for correct, 0 otherwise).",
    )
    # Note: We might add more reward configuration parameters here later,
    # e.g., similarity_metric, thresholds, penalty weights.

    # Nested Training Configuration
    training_config: Optional[TrainingConfig] = Field(
        default=None,
        description="Training-specific hyperparameters. Only used for trainable models.",
    )


# Example Usage (in train.py):
# agent_config = ProjectPolicyConfig(
#     max_turns=15,
#     use_tools=True,
#     training_config=TrainingConfig(
#         learning_rate=1e-5,
#         num_epochs=2,
#         # ... other training params
#     )
# )
#
# benchmark_config = ProjectPolicyConfig(
#     litellm_model_name="openai/gpt-4o",
#     use_tools=True,
#     # training_config is None for non-trainable models
# )
