# cuad_qa/rollout.py (Corrected with LLM Judge)

import art
import json
import logging
import os
import textwrap

from typing import List, Any, Optional, Dict, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# ART and LLM imports
from art import Trajectory
from art.utils import limit_concurrency
from art.utils.litellm import convert_litellm_choice_to_openai
import litellm
from litellm.caching.caching import LiteLLMCacheType, Cache
from litellm.types.utils import Choices, ModelResponse, Message
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from langchain_core.utils.function_calling import convert_to_openai_tool
from tenacity import retry, stop_after_attempt  # <<< Re-added for LLM judge call

# Project specific imports
from cuad_qa.data.generate_cuad_scenarios import ClauseFindingScenario
from cuad_qa.contract_search_tools import (
    search_contracts,
    read_contract,
    ContractSearchResult,
)
from cuad_qa.project_configs import ProjectPolicyConfig

# Optional OpenPipe logging (Keep commented out unless used)
# from openpipe import AsyncOpenPipe
# if os.getenv("OPENPIPE_API_KEY"):
#     op_client = AsyncOpenPipe()
# else:
#     op_client = None

# Configure LiteLLM caching
litellm.cache = Cache(type=LiteLLMCacheType.DISK)
# litellm._turn_on_debug()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def return_final_answer(extracted_clause_text: str, source_contract_id: int) -> str:
    """
    Call this function to return the exact text of the clause you found.
    If you cannot find the clause, call this with "I could not find the clause." as the text and -1 as the source_contract_id.

    Args:
        extracted_clause_text (str): The exact text of the clause found in the contract. If not found, use "I could not find the clause.".
        source_contract_id (int): The ID of the contract where the clause was found. Use -1 if not found.

    Returns:
        str: Confirmation message indicating the answer was recorded.
    """
    return f"Final answer recorded: {extracted_clause_text[:50]}..."


# --- Tool Definitions ---
final_answer_tool_def = convert_to_openai_tool(return_final_answer)
search_tool_def = convert_to_openai_tool(search_contracts)
read_tool_def = convert_to_openai_tool(read_contract)

tools: list[ChatCompletionToolParam] = [
    search_tool_def,
    read_tool_def,
    final_answer_tool_def,
]  # type: ignore


# --- Rubric Definition ---
@dataclass
class FinalRubric:
    """
    - Dimensions on which a single agent attempt (a rollout) will be judged.
    Tracks metrics during a single rollout for reward calculation.

    - Once the rollout terminates (either by the agent calling return_final_answer,
    running out of turns, or encountering a critical error), this FinalRubric object
    holds the complete picture of what happened during that single attempt.

    - This finalized set of criteria is then passed to the calculate_reward function to
    determine the single scalar reward value for that entire trajectory.
    """

    clause_text_correct: bool = False
    found_correct_contract: bool = False
    num_turns: int = 0
    attempted_answer: bool = False
    returned_i_dont_know: bool = False
    ever_found_correct_contract_in_search: bool = False
    ever_read_correct_contract: bool = False
    tried_to_read_invalid_contract: bool = False
    cant_parse_tool_call: bool = False
    bad_tool_call_name: bool = False
    bad_tool_call_args: bool = False
    ran_out_of_turns: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    judge_prompt_tokens: int = 0  # <<< ADDED: Track judge cost separately
    judge_completion_tokens: int = 0  # <<< ADDED: Track judge cost separately

    def to_metrics(self) -> dict[str, float | int | bool | None]:
        """Converts rubric to a dictionary suitable for logging."""
        metric_dict = {}
        for k, v in asdict(self).items():
            if isinstance(v, bool):
                metric_dict[k] = int(v)
            else:
                metric_dict[k] = v
        return metric_dict


# --- LLM Judge for Correctness ---
# Added retry decorator for robustness against temporary API issues
@retry(stop=stop_after_attempt(3), wait=litellm.utils.retry_with_exponential_backoff)
async def determine_clause_correctness_with_llm(
    extracted_text: str,
    scenario: ClauseFindingScenario,
    rubric: FinalRubric,  # Pass rubric to track token usage
) -> bool:
    """
    Uses an LLM judge to determine if the extracted clause text semantically
    matches the ground truth text for the given clause type.

    Args:
        extracted_text: The clause text extracted by the agent.
        scenario: The ClauseFindingScenario containing ground truth.
        rubric: The FinalRubric object to update token counts.

    Returns:
        True if the extracted text is considered a correct match, False otherwise.
    """
    judge_model = "gpt-4.1-nano"

    # Robust check for empty strings
    if not extracted_text or not scenario.ground_truth_clause_text:
        logging.warning(
            "Cannot judge correctness: Extracted or ground truth text is empty."
        )
        return False

    # Clear prompt asking for a boolean judgment
    system_prompt = textwrap.dedent("""\
        You are an expert evaluator comparing extracted legal clauses. You will see the target clause type, the ground truth text, and the text extracted by an AI assistant.
        Determine if the Extracted Text accurately and completely represents the Ground Truth Text for the specified Clause Type.
        Ignore minor differences in whitespace or punctuation unless they change the meaning significantly.
        Focus on semantic equivalence and completeness.
        Respond ONLY with the word 'True' if the extracted text is a good match, or 'False' otherwise. No other text, reasoning, or punctuation.
        """)

    user_content = (
        f"Clause Type: {scenario.clause_type}\n\n"
        f"Ground Truth Text:\n```\n{scenario.ground_truth_clause_text}\n```\n\n"
        f"Extracted Text:\n```\n{extracted_text}\n```\n\n"
        f"Is the Extracted Text a correct match for the Ground Truth Text given the Clause Type? Respond only True or False."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    try:
        response = await litellm.acompletion(
            model=judge_model,
            messages=messages,
            temperature=0.0,  # Low temperature for deterministic judgment
            max_tokens=5,  # Expect short response ("True" or "False")
            # Caching is useful for deterministic evals, especially if retrying
            caching=True,
            # Make sure API keys are available if needed (e.g., in .env)
            api_key=os.getenv("OPENAI_API_KEY"),  # Adjust if using different provider
        )

        # Track judge token usage
        if response.usage:
            rubric.judge_prompt_tokens += response.usage.prompt_tokens or 0
            rubric.judge_completion_tokens += response.usage.completion_tokens or 0

        # Robust parsing of the response
        llm_response_content = response.choices[0].message.content
        if llm_response_content:
            result = llm_response_content.strip().lower().startswith("t")
            logging.debug(
                f"LLM Judge response for scenario {scenario.scenario_id}: '{llm_response_content}' -> Parsed as {result}"
            )
            return result
        else:
            logging.warning(
                f"LLM Judge returned empty content for scenario {scenario.scenario_id}."
            )
            return False

    except Exception as e:
        logging.error(
            f"Error calling LLM judge for scenario {scenario.scenario_id}: {e}"
        )
        # Default to False if the judge fails
        return False


# --- Reward Calculation ---
def calculate_reward(
    policy_config: ProjectPolicyConfig, rubric: FinalRubric, traj: Trajectory
) -> float:
    """Calculates the reward for a completed trajectory based on the rubric."""
    # --- Penalty Section (Applied regardless of final answer) ---
    if rubric.cant_parse_tool_call:
        return -2.0
    if rubric.bad_tool_call_name:
        return -1.9
    if rubric.bad_tool_call_args:
        return -1.8
    if rubric.tried_to_read_invalid_contract:
        return -1.7

    # --- Outcome-Based Reward/Penalty Section ---
    if not rubric.attempted_answer:
        base_reward = -0.5 if rubric.ran_out_of_turns else -0.3
        partial_credit = 0.0
        partial_credit += 0.05 if rubric.ever_found_correct_contract_in_search else 0
        partial_credit += 0.05 if rubric.ever_read_correct_contract else 0
        return base_reward + partial_credit

    if rubric.attempted_answer:
        if rubric.clause_text_correct and rubric.found_correct_contract:
            reward = 1.0
            efficiency_bonus = 0.2 * (1 - rubric.num_turns / policy_config.max_turns)
            reward += efficiency_bonus
            return reward
        else:
            base_penalty = -1.0 if rubric.found_correct_contract else -1.2
            partial_credit = 0.0
            partial_credit += (
                0.05 if rubric.ever_found_correct_contract_in_search else 0
            )
            partial_credit += 0.05 if rubric.ever_read_correct_contract else 0
            # Add a small partial credit based on the *judge's* score if text was wrong but contract right
            # This part was removed as the judge only returns boolean now. Keep it simple.
            return base_penalty + partial_credit

    logging.error(f"Unhandled rubric state: {rubric}")
    traj.logs.append(f"Rubric not handled properly: {rubric}")
    return -3.0


# --- Helper for Tool Response Formatting ---
def tool_response(response: Any, tool_call_id: str) -> ChatCompletionMessageParam:
    """Generate a response message for a tool call."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(response),
    }


# --- Main Rollout Function ---
@limit_concurrency(10, derive_key=lambda model, scenario, **kwargs: model.name)
async def rollout(
    model: art.Model,
    scenario: ClauseFindingScenario,
) -> Trajectory:
    """Executes a single agent interaction loop for a ClauseFindingScenario."""
    rollout_start_time = datetime.now()
    rubric = FinalRubric()
    traj = Trajectory(
        messages_and_choices=[],
        reward=0,
        metadata={
            "scenario_id": scenario.scenario_id,
            "contract_id": scenario.contract_id,
            "contract_title": scenario.contract_title,
            "target_clause_type": scenario.clause_type,
            "annotation_id": scenario.annotation_id,
        },
    )
    assert isinstance(model.config, ProjectPolicyConfig)

    # --- Construct System Prompt ---
    system_prompt = textwrap.dedent(f"""\
        You are an expert legal assistant AI. Your task is to find specific clauses in legal contracts.
        You will be given the type of clause to find. Use the provided tools to search contracts and read their content to locate the requested clause text.

        Available Tools:
        - `search_contracts(keywords: list[str], max_results: int)`: Searches contracts by keywords in title or text. Returns contract IDs, titles, and snippets.
        - `read_contract(contract_id: int)`: Reads the full text of a specific contract given its ID.
        - `return_final_answer(extracted_clause_text: str, source_contract_id: int)`: Use this ONLY when you have found the exact text of the requested clause. Provide the full clause text and the ID of the contract it came from. If you cannot find the clause after searching, use this tool with "I could not find the clause." and source_contract_id=-1.

        Current Task: Find the clause text for the following clause type in contract '{scenario.contract_title}' (ID: {scenario.contract_id}):
        Clause Type: "{scenario.clause_type}"

        Think step-by-step. First, search for the contract if necessary, then read it, then identify the exact clause text based on the type, and finally return the answer using `return_final_answer`. You have a maximum of {model.config.max_turns} turns.
    """)

    if not model.config.use_tools:
        system_prompt += '\n\n Respond with JSON for tool calls: {"tool_name": "...", "tool_args": {...}}'

    if model.config.use_tools:
        traj.tools = tools

    traj.messages_and_choices = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Please find the '{scenario.clause_type}' clause in contract ID {scenario.contract_id}.",
        },
    ]

    llm_response: Optional[ModelResponse] = None
    extracted_answer_text: Optional[str] = None
    final_source_contract_id: Optional[int] = None

    # --- Agent Interaction Loop ---
    while True:
        rubric.num_turns += 1

        if rubric.num_turns > model.config.max_turns:
            logging.warning(
                f"Scenario {scenario.scenario_id}: Ran out of turns ({model.config.max_turns})."
            )
            rubric.ran_out_of_turns = True
            break

        litellm_model_name = model.config.litellm_model_name
        if litellm_model_name is None and model.trainable:
            litellm_model_name = f"hosted_vllm/{model.name}"
        elif litellm_model_name is None:
            raise ValueError(
                "litellm_model_name must be set in ProjectPolicyConfig for non-trainable models"
            )

        # --- Call LLM for next action ---
        try:
            # (LLM Call logic - remains the same as previous version)
            llm_response = await litellm.acompletion(
                model=litellm_model_name,
                base_url=model.base_url,
                messages=traj.messages(),
                caching=not model.trainable,
                api_key=model.api_key,
                max_tokens=model.config.max_tokens,
                tools=tools if model.config.use_tools else None,
                tool_choice="auto" if model.config.use_tools else None,
            )
            assert isinstance(
                llm_response, ModelResponse
            ), "LLM response is not a ModelResponse"

        except Exception as e:
            logging.error(
                f"LLM API call failed for scenario {scenario.scenario_id}: {e}"
            )
            rubric.bad_tool_call_name = True  # Treat as failure
            break

        # Update cost metrics (Agent LLM)
        if llm_response.usage:
            rubric.prompt_tokens += llm_response.usage.prompt_tokens or 0
            rubric.completion_tokens += llm_response.usage.completion_tokens or 0

        choice = llm_response.choices[0]
        assert isinstance(choice, Choices), "LLM choice is not valid"

        # Append LLM response to trajectory
        if model.trainable:
            traj.messages_and_choices.append(convert_litellm_choice_to_openai(choice))
        else:
            traj.messages_and_choices.append(choice.message.to_dict())

        # --- Parse and Execute Tool Call ---
        tool_name = None
        tool_args = None
        tool_call_id = None

        # Handle OpenAI tool format
        if model.config.use_tools and choice.message.tool_calls:
            # Process only the first tool call if multiple are generated
            if len(choice.message.tool_calls) > 1:
                logging.warning(
                    f"Scenario {scenario.scenario_id}: Model generated multiple tool calls, using only the first."
                )
            first_tool_call = choice.message.tool_calls[0]
            tool_call_id = first_tool_call.id
            if first_tool_call.function:
                tool_name = first_tool_call.function.name
                try:
                    tool_args = json.loads(first_tool_call.function.arguments)
                    assert isinstance(tool_args, dict)
                except (json.JSONDecodeError, AssertionError) as e:
                    logging.warning(
                        f"Scenario {scenario.scenario_id}: Failed to parse tool arguments: {e}. Args: {first_tool_call.function.arguments}"
                    )
                    rubric.bad_tool_call_args = True
                    break  # Abort on bad args
            else:
                logging.warning(
                    f"Scenario {scenario.scenario_id}: Tool call missing function details."
                )
                rubric.bad_tool_call_args = True
                break

        # Handle JSON format (if use_tools is False)
        elif not model.config.use_tools and choice.message.content:
            raw_content = choice.message.content
            try:
                # Basic JSON object extraction
                start_index = raw_content.find("{")
                end_index = raw_content.rfind("}")
                if start_index != -1 and end_index != -1 and start_index < end_index:
                    json_str = raw_content[start_index : end_index + 1]
                    parsed_call = json.loads(json_str)
                    if (
                        isinstance(parsed_call, dict)
                        and "tool_name" in parsed_call
                        and "tool_args" in parsed_call
                    ):
                        tool_name = parsed_call.get("tool_name")
                        tool_args = parsed_call.get("tool_args")
                        # Note: No tool_call_id in this format
                    else:
                        raise ValueError("JSON missing 'tool_name' or 'tool_args'")
                else:
                    raise ValueError("Could not find valid JSON object in content")
            except Exception as e:
                logging.warning(
                    f"Scenario {scenario.scenario_id}: Failed to parse JSON tool call: {e}. Content: {raw_content}"
                )
                rubric.cant_parse_tool_call = True
                break  # Abort on bad parse

        # If no tool call was made by the LLM (e.g., just text response when tool expected)
        if tool_name is None or tool_args is None:
            # Check if it was *supposed* to be the final answer turn but didn't use the tool
            if (
                choice.message.content and not model.config.use_tools
            ):  # Agent responded text instead of JSON tool call
                logging.warning(
                    f"Scenario {scenario.scenario_id}: Agent responded with text instead of tool JSON. Treating as failed attempt."
                )
                # We could try to interpret this as the final answer, but it violates the expected format. Penalize.
                rubric.cant_parse_tool_call = True
                break
            elif (
                choice.message.content and model.config.use_tools
            ):  # Agent responded text instead of tool call
                logging.warning(
                    f"Scenario {scenario.scenario_id}: Agent responded with text instead of tool call. Treating as failed attempt."
                )
                rubric.bad_tool_call_name = True  # Or a different rubric flag?
                break
            else:  # No tool call and no content
                logging.warning(
                    f"Scenario {scenario.scenario_id}: LLM did not call a tool or return content."
                )
                rubric.bad_tool_call_name = True  # Treat as if it failed to call a tool
                break

        # --- Execute the selected tool ---
        tool_result_content = None
        try:
            match tool_name:
                case "search_contracts":
                    # (search_contracts logic - keep as before)
                    search_results = search_contracts(**tool_args)
                    for res in search_results:
                        if res.contract_id == scenario.contract_id:
                            rubric.ever_found_correct_contract_in_search = True
                            break
                    tool_result_content = [asdict(r) for r in search_results]

                case "read_contract":
                    # (read_contract logic - keep as before)
                    contract_id_to_read = tool_args.get("contract_id")
                    if not isinstance(contract_id_to_read, int):
                        rubric.bad_tool_call_args = True
                        break
                    if contract_id_to_read <= 0:
                        rubric.tried_to_read_invalid_contract = True
                        tool_result_content = {
                            "error": f"Contract ID {contract_id_to_read} is invalid."
                        }
                    else:
                        if contract_id_to_read == scenario.contract_id:
                            rubric.ever_read_correct_contract = True
                        contract_data = read_contract(contract_id_to_read)
                        if contract_data is None:
                            rubric.tried_to_read_invalid_contract = True
                            tool_result_content = {
                                "error": f"Contract with ID {contract_id_to_read} not found."
                            }
                        else:
                            tool_result_content = contract_data

                case "return_final_answer":
                    extracted_answer_text = tool_args.get("extracted_clause_text")
                    final_source_contract_id = tool_args.get("source_contract_id")

                    if not isinstance(extracted_answer_text, str) or not isinstance(
                        final_source_contract_id, int
                    ):
                        logging.warning(
                            f"Scenario {scenario.scenario_id}: Invalid args type for return_final_answer: {tool_args}"
                        )
                        rubric.bad_tool_call_args = True
                        break

                    rubric.attempted_answer = True
                    if (
                        extracted_answer_text.strip().lower()
                        == "i could not find the clause."
                        or final_source_contract_id == -1
                    ):
                        rubric.returned_i_dont_know = True
                        rubric.clause_text_correct = False
                        rubric.found_correct_contract = False
                    else:
                        # <<< MODIFIED: Use LLM Judge >>>
                        is_correct = await determine_clause_correctness_with_llm(
                            extracted_answer_text,
                            scenario,
                            rubric,  # Pass rubric for token counts
                        )
                        rubric.clause_text_correct = is_correct
                        # Set similarity score based on boolean judge result
                        # rubric.answer_similarity_score = 1.0 if is_correct else 0.0 # Removed from rubric
                        rubric.found_correct_contract = (
                            final_source_contract_id == scenario.contract_id
                        )
                        # <<< END MODIFICATION >>>

                    logging.info(
                        f"Scenario {scenario.scenario_id}: Agent returned final answer. Correct Text: {rubric.clause_text_correct}, Correct Contract: {rubric.found_correct_contract}"
                    )
                    break  # EXIT LOOP

                case _:
                    logging.warning(
                        f"Scenario {scenario.scenario_id}: Agent called unknown tool: {tool_name}"
                    )
                    rubric.bad_tool_call_name = True
                    break

        # (Error handling for tool execution - keep as before)
        except TypeError as e:
            logging.warning(
                f"Scenario {scenario.scenario_id}: TypeError calling tool '{tool_name}' with args {tool_args}. Error: {e}"
            )
            rubric.bad_tool_call_args = True
            break
        except Exception as e:
            logging.error(
                f"Scenario {scenario.scenario_id}: Unexpected error executing tool '{tool_name}'. Error: {e}"
            )
            rubric.bad_tool_call_args = True
            break

        # --- Append Tool Response ---
        if tool_result_content is not None and tool_call_id is not None:
            traj.messages_and_choices.append(
                tool_response(tool_result_content, tool_call_id)
            )
        # (Handle other cases - keep as before)
        elif tool_result_content is not None and not model.config.use_tools:
            logging.error("JSON tool response handling not fully implemented yet.")
            break
        else:
            logging.error(
                f"Scenario {scenario.scenario_id}: Reached end of loop unexpectedly after tool call {tool_name}."
            )
            break

    # --- Finalize Trajectory ---
    # (Finalization logic - keep as before)
    reward = calculate_reward(model.config, rubric, traj)
    traj.reward = reward
    traj.metrics = rubric.to_metrics()
    rollout_end_time = datetime.now()
    duration_seconds = (rollout_end_time - rollout_start_time).total_seconds()
    traj.metrics["duration"] = duration_seconds
    logging.info(
        f"Finished rollout for scenario {scenario.scenario_id} in {duration_seconds:.2f}s. Reward: {reward:.2f}. Turns: {rubric.num_turns}. Correct: {rubric.clause_text_correct}"
    )

    return traj


# --- Example Usage (for testing) ---
if __name__ == "__main__":
    # (Keep the testing block mostly the same, but note the need for judge LLM API key)
    from cuad_qa.data.query_iterators import load_clause_finding_scenarios
    from cuad_qa.project_configs import ProjectPolicyConfig, TrainingConfig
    import asyncio
    import yaml
    from dotenv import load_dotenv

    load_dotenv()

    async def test_rollout():
        print("--- Testing CUAD Rollout (with LLM Judge) ---")
        # (Load scenario logic - keep as before)
        try:
            if not os.path.exists(DEFAULT_CUAD_DB_PATH):
                print(f"Error: DB not found at {DEFAULT_CUAD_DB_PATH}")
                return
            test_scenarios = load_clause_finding_scenarios(split="test", limit=1)
            if not test_scenarios:
                print("Error: Could not load test scenarios.")
                return
            scenario = test_scenarios[0]
            print(
                f"Using Scenario ID: {scenario.scenario_id}, Contract ID: {scenario.contract_id}"
            )
            print(f"Target Clause Type: {scenario.clause_type}")
            print(
                f"Ground Truth Text (first 100 chars): {scenario.ground_truth_clause_text[:100]}..."
            )
        except Exception as e:
            print(f"Error loading test scenario: {e}")
            return

        # NOTE: Testing now requires API key for the JUDGE LLM (e.g., GOOGLE_API_KEY for Gemini Flash)
        # AND potentially an API key for the AGENT LLM if testing an external one.
        # Ensure keys are in your .env file.
        agent_api_key = os.getenv("OPENAI_API_KEY")  # Example for agent
        judge_api_key = os.getenv("GOOGLE_API_KEY")  # Example for judge

        if not agent_api_key:
            print("Warning: OPENAI_API_KEY not found for agent model test.")
        if not judge_api_key:
            print("Warning: GOOGLE_API_KEY not found for judge model test.")

        model_to_test = art.Model(
            name="gpt-4o-mini",
            project="cuad_agent_test",
            api_key=agent_api_key,
            config=ProjectPolicyConfig(
                litellm_model_name="openai/gpt-4o-mini", use_tools=True, max_turns=8
            ),
        )

        print(f"\nRunning rollout with model: {model_to_test.name}...")
        try:
            trajectory = await rollout(model_to_test, scenario)
            # (Print results logic - keep as before)
            print("\n--- Rollout Complete ---")
            print(f"Final Reward: {trajectory.reward}")
            print("Metrics:")
            for k, v in trajectory.metrics.items():
                print(f"  {k}: {v}")
            print("\nTrajectory Log (YAML):")
            try:
                print(yaml.dump(trajectory.for_logging()))
            except Exception as dump_error:
                print(f"Error dumping: {dump_error}")
                print("Messages:", trajectory.messages_and_choices)
        except Exception as e:
            print(f"\nError during rollout execution: {e}")

    asyncio.run(test_rollout())
