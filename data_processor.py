import json
import os
import logging
from typing import Dict, List, Tuple
import re

from models import Message, ModelBase

"""
DataProcessor should focus solely on the training aspects:

Generating code from prompts
Comparing generated code with reference code (debug analysis)
Using these comparisons to improve the instruction prompt
"""


class BatchProcessingError(Exception):
    pass


class CodeGenerationError(Exception):
    pass


class DataProcessor:

    def __init__(self, DEBUG_PROMPT: str, OPTIMIZE_PROMPT: str):
        self.logger = logging.getLogger("experiment.data_processor")
        self.DEBUG_PROMPT = DEBUG_PROMPT
        self.OPTIMIZE_PROMPT = OPTIMIZE_PROMPT
        self.logger.debug("DataProcessor initialized with debug and optimize prompts")

    def load_dataset(self, dataset_path: str, file_name: str) -> List[Dict]:
        full_path = os.path.join(dataset_path, file_name)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File `{full_path}` does not exist.")
        with open(full_path, "r") as file:
            return json.load(file)

    def process_batch(
        self,
        batch: List[Dict],
        model: ModelBase,
        args,
        current_instruction: str,
    ) -> Tuple[List[Dict], Dict, str]:
        """Processes a batch of tasks, generating code and analyzing results."""
        self.logger.debug(f"Processing batch of {len(batch)} items")
        batch_results = []

        for item in batch:
            self.logger.debug(f"Processing task ID: {item['task_id']}")
            task_id = str(item["task_id"])
            prompt = item["prompt"]
            reference_code = item["code"]

            try:
                self.logger.debug(f"Generating code for task {task_id}")
                generated_code = self.generate(
                    prompt,
                    model,
                    args.temperature,
                    args.seed,
                    current_instruction,
                )

                # Clean the generated code
                generated_code = (
                    generated_code.strip("`").replace("python\n", "").strip()
                )

                # Extract function name from generated code
                func_name = self._extract_function_name(generated_code)
                # self.logger.debug(f"Extracted function name: {func_name} for task {task_id}")

                result = {
                    "task_id": task_id,
                    "generated_code": generated_code,
                    "reference_code": reference_code,
                    "func_name": func_name,
                    "test": item["test"],
                }
                batch_results.append(result)
                self.logger.debug(f"Successfully processed task {task_id}")

            except Exception as e:
                error_msg = f"Error in code generation: {str(e)}"
                self.logger.error(error_msg)
                raise BatchProcessingError(error_msg)

        self.logger.debug(
            f"Completed processing of {len(batch_results)} items in batch"
        )

        try:
            self.logger.debug("Starting batch analysis")
            batch_analysis = self.debug(
                batch_results, model, args.temperature, args.seed
            )
            self.logger.debug("Batch analysis completed")

            self.logger.debug("Starting instruction optimization")
            updated_instruction = self.optimize(
                current_instruction,
                batch_analysis["comparisons"],
                model,
                args.temperature,
                args.seed,
            )
            self.logger.debug("Instruction optimization completed")

        except Exception as e:
            error_msg = f"Error in debug analysis: {str(e)}"
            self.logger.error(error_msg)
            raise BatchProcessingError(error_msg)

        self.logger.info("Batch processing completed")
        return batch_results, batch_analysis, updated_instruction

    def generate(
        self,
        prompt: str,
        model: ModelBase,
        temperature: float,
        seed: int,
        current_instruction: str,
    ) -> str:
        """Generate code using the generator agent."""
        messages = [
            Message(role="system", content=current_instruction),
            Message(
                role="user",
                content=f"Generate a Python function based on:\n{prompt}",
            ),
        ]

        try:
            generated_code = model.generate_chat(
                messages=messages, temperature=temperature, seed=seed
            )
            return generated_code

        except Exception as e:
            self.logger.error(f"Code generation failed: {str(e)}")
            raise CodeGenerationError(
                f"An error occurred during code generation: {str(e)}"
            )

    def debug(
        self,
        code_pairs: List[Dict],
        model: ModelBase,
        temperature: float,
        seed: int,
    ) -> Dict[str, str]:
        """Debug and analyze code pairs using the debugger agent."""
        self.logger.debug(f"debugging - this batch has {len(code_pairs)} code pairs")

        try:
            debug_results = []
            for pair in code_pairs:
                debug_messages = [
                    Message(role="system", content=self.DEBUG_PROMPT),
                    Message(role="user", content=f"Task ID: {pair['task_id']}"),
                    Message(
                        role="user",
                        content=f"Generated Code:\n{pair['generated_code'].strip('`').strip()}",
                    ),
                    Message(
                        role="user",
                        content=f"Reference Code:\n{pair['reference_code']}",
                    ),
                ]

                debug_result = model.generate_chat(
                    messages=debug_messages, temperature=temperature, seed=seed
                )
                debug_results.append(debug_result)

            return {"comparisons": debug_results}

        except Exception as e:
            self.logger.error(f"Error in debug analysis: {str(e)}")
            return {"comparisons": []}

    def optimize(
        self,
        current_instruction: str,
        batch_analysis: str,
        model: ModelBase,
        temperature: float,
        seed: int,
    ) -> str:
        """Optimize instructions using the optimizer agent."""
        messages = [
            Message(role="system", content=self.OPTIMIZE_PROMPT),
            Message(
                role="user",
                content=f"Current instruction:\n\n{current_instruction}",
            ),
            Message(role="user", content=f"Comparisons:\n\n{batch_analysis}"),
        ]

        try:
            optimized_instruction = model.generate_chat(
                messages=messages, temperature=temperature, seed=seed
            )
            return optimized_instruction

        except Exception as e:
            self.logger.error(f"Error in instruction optimization: {str(e)}")
            return current_instruction

    def _extract_function_name(self, code: str) -> str:
        """Extract function name from generated code."""
        try:
            # Match 'def function_name(' pattern
            match = re.search(r"def\s+([a-zA-Z_]\w*)\s*\(", code)
            if match:
                return match.group(1)
            return "unknown_function"
        except Exception as e:
            self.logger.error(f"Error extracting function name: {str(e)}")
            return "unknown_function"
