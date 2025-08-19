import os
from typing import Dict, List
import json
import logging

from models import Message, ModelBase
from executor import Executor, ExecuteResult
from metrics import RunMetrics


class DevEvaluation:
    """
    Handles evaluation of instructions on the development dataset.
    Works in conjunction with Executor for running tests and RunMetrics for tracking performance.
    """

    def __init__(
        self,
        model: ModelBase,
        dev_data: List[Dict],
        executor: Executor,
        run_metrics: RunMetrics,
        experiment_dir: str,
    ):
        """Initialize DevEvaluation with necessary components."""
        self.model = model
        self.dev_data = dev_data
        self.executor = executor  # Uses existing Executor instance
        self.run_metrics = run_metrics  # Uses existing RunMetrics instance
        self.evaluation_dir = os.path.join(experiment_dir, "dev_evaluation")
        os.makedirs(self.evaluation_dir, exist_ok=True)

    def evaluate_instruction(self, instruction: str, epoch: int, batch: int) -> Dict:
        """
        Evaluates an instruction's performance on the entire dev set.
        Creates a comprehensive record of the evaluation, including individual task results
        and overall performance metrics.
        """
        logger = logging.getLogger("experiment.evaluation")
        logger.info(f"Evaluating instruction for epoch {epoch}, batch {batch}")
        batch_dir = os.path.join(
            self.evaluation_dir, f"epoch_{epoch}", f"batch_{batch}"
        )
        os.makedirs(os.path.join(batch_dir, "individual_tasks"), exist_ok=True)

        dev_results = []

        for item in self.dev_data:
            # Generate code using current instruction
            messages = [
                Message(role="system", content=instruction),
                Message(
                    role="user",
                    content=f"Generate a Python function based on:\n{item['prompt']}",
                ),
            ]
            generated_code = self.model.generate_chat(messages=messages)

            # Use existing Executor instance to get ExecuteResult
            execution_result: ExecuteResult = self.executor.execute(
                generated_code, item["test"]
            )

            task_result = {
                "task_id": item["task_id"],
                "generated_code": generated_code,
                "reference_code": item["code"],
                "execution_results": {
                    "is_passing": execution_result.is_passing,
                    "passed_tests": execution_result.passed_tests,
                    "failed_tests": execution_result.failed_tests,
                },
            }

            # Save individual task result
            task_path = os.path.join(
                batch_dir, "individual_tasks", f"task_{item['task_id']}.json"
            )
            with open(task_path, "w") as f:
                json.dump(task_result, f, indent=2)

            dev_results.append(task_result)

        # Calculate overall pass rate
        overall_pass_rate = sum(
            1 for r in dev_results if r["execution_results"]["is_passing"]
        ) / len(dev_results)

        # Update RunMetrics with dev set performance
        self.run_metrics.update(epoch, batch, overall_pass_rate)

        evaluation_summary = {
            "instruction_evaluated": instruction,
            "overall_pass_rate": overall_pass_rate,
            "detailed_results": dev_results,
        }

        # Save combined evaluation results
        with open(os.path.join(batch_dir, "dev_evaluation.json"), "w") as f:
            json.dump(evaluation_summary, f, indent=2)

        logger.info(f"Dev evaluation - Pass rate: {overall_pass_rate:.2%}\n")
        return evaluation_summary
