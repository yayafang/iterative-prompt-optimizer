import argparse
import os
import sys
import logging
from typing import Dict, List
import json
import math

from models import ModelBase, get_model
from data_processor import DataProcessor
from executor import Executor
from metrics import RunMetrics
from dev_eval import DevEvaluation


def setup_imports():
    from prompts.generator import GENERATE_PROMPT
    from prompts.debugger import DEBUG_PROMPT
    from prompts.optimizer import OPTIMIZE_PROMPT

    return (GENERATE_PROMPT, DEBUG_PROMPT, OPTIMIZE_PROMPT)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Required arguments
    required = parser.add_argument_group("required arguments")
    required.add_argument("--experiment_name", type=str, required=True)
    required.add_argument("--run_name", type=str, required=True)
    required.add_argument("--root_dir", type=str, required=True)
    required.add_argument("--dataset_path", type=str, required=True)
    required.add_argument("--train_file", type=str, required=True)
    required.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["gpt4-turbo", "gpt-4o", "gpt-35-turbo"],
    )
    required.add_argument("--seed", type=int, default=42)
    required.add_argument("--temperature", type=float, default=0.0)
    required.add_argument("--batch_size", type=int, default=10)
    required.add_argument("--epochs", type=int, default=1)
    required.add_argument("--dev_file", type=str)

    # Optional arguments
    parser.add_argument("--test_file", type=str)

    args = parser.parse_args()

    return args


def load_checkpoint(checkpoint_path: str) -> Dict:
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    return {}


def save_checkpoint(
    args, epoch_num, batch_num, current_instruction, updated_instruction
):
    checkpoint_data = {
        "last_completed_epoch": epoch_num,
        "last_completed_batch": batch_num,
        "current_prompt": current_instruction,  # prompt used in this batch, what we used in batch_num
        "next_prompt": updated_instruction,  # prompt to use in next batch,  what we'll use in batch_num+1
    }
    with open(args.checkpoint_path, "w") as f:
        json.dump(checkpoint_data, f, indent=2)


def setup_directories(args: argparse.Namespace) -> None:
    """
    Sets up the directory structure for the experiment, organizing it into
    training and evaluation sections.
    """
    # Create main experiment directory
    experiment_dir = os.path.join(args.root_dir, args.experiment_name)

    # Create training-related directories
    training_dir = os.path.join(experiment_dir, "training")
    training_epochs_dir = os.path.join(training_dir, "epochs")
    checkpoint_dir = os.path.join(training_dir, "checkpoints")

    # Create evaluation-related directory
    dev_eval_dir = os.path.join(experiment_dir, "dev_evaluation")

    # Create all directories
    os.makedirs(training_epochs_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Store paths in args for later use
    args.experiment_dir = experiment_dir
    args.training_dir = training_dir
    args.checkpoint_dir = checkpoint_dir
    args.dev_eval_dir = dev_eval_dir

    args.checkpoint_path = os.path.join(args.checkpoint_dir, "latest.json")


def setup_logging(args: argparse.Namespace) -> logging.Logger:
    log_dir = os.path.join(args.experiment_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"experiment.log")

    # File handler - set to DEBUG to capture all levels
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    # Console handler - keep at INFO to show only important messages
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    # Configure root logger at DEBUG level to allow all messages
    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])
    logger = logging.getLogger("experiment")

    # Confgure openai
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Only log basic experiment info to console
    logging.info(
        f"\nStarting experiment: {args.experiment_name} (Run: {args.run_name})\nModel: {args.model}, Batch size: {args.batch_size}, Epochs: {args.epochs}\n"
    )

    return logger


def create_dev_summary(args):
    """Create a summary of dev evaluation results across all batches."""
    dev_summary = {
        "experiment_name": args.experiment_name,
        "run_name": args.run_name,
        "batch_metrics": [],
    }

    for epoch_dir in sorted(os.listdir(args.dev_eval_dir)):
        if not epoch_dir.startswith("epoch_"):
            continue
        epoch_num = int(epoch_dir.split("_")[1])

        epoch_path = os.path.join(args.dev_eval_dir, epoch_dir)
        for batch_dir in sorted(os.listdir(epoch_path)):
            if not batch_dir.startswith("batch_"):
                continue
            batch_num = int(batch_dir.split("_")[1])

            eval_path = os.path.join(epoch_path, batch_dir, "dev_evaluation.json")
            if os.path.exists(eval_path):
                with open(eval_path, "r") as f:
                    eval_data = json.load(f)

                batch_summary = {
                    "epoch": epoch_num,
                    "batch": batch_num,
                    "pass_rate": eval_data["overall_pass_rate"],
                    "instruction": eval_data["instruction_evaluated"],
                }
                dev_summary["batch_metrics"].append(batch_summary)

    # Save summary
    summary_path = os.path.join(args.dev_eval_dir, "dev_overall.json")
    with open(summary_path, "w") as f:
        json.dump(dev_summary, f, indent=2)

    return dev_summary


def track_prompt_evolution(args):
    """Create a summary of prompt evolution across batches."""
    prompt_evolution = {
        "experiment_name": args.experiment_name,
        "run_name": args.run_name,
        "prompts": [],
    }

    for epoch_dir in sorted(os.listdir(os.path.join(args.training_dir, "epochs"))):
        if not epoch_dir.startswith("epoch_"):
            continue
        epoch_num = int(epoch_dir.split("_")[1])

        epoch_path = os.path.join(args.training_dir, "epochs", epoch_dir)
        for batch_dir in sorted(os.listdir(epoch_path)):
            if not batch_dir.startswith("batch_"):
                continue
            batch_num = int(batch_dir.split("_")[1])

            results_path = os.path.join(epoch_path, batch_dir, "batch_results.json")
            if os.path.exists(results_path):
                with open(results_path, "r") as f:
                    batch_data = json.load(f)

                prompt_data = {
                    "epoch": epoch_num,
                    "batch": batch_num,
                    "current_prompt": batch_data["instruction"]["current"],
                    "updated_prompt": batch_data["instruction"]["updated"],
                }
                prompt_evolution["prompts"].append(prompt_data)

    # Save summary
    summary_path = os.path.join(args.training_dir, "prompt_evolution.json")
    with open(summary_path, "w") as f:
        json.dump(prompt_evolution, f, indent=2)

    return prompt_evolution


def main():
    """
    Main entry point of the program. Sets up the experiment environment,
    initializes components, and runs the training loop.
    """
    args = parse_arguments()
    setup_directories(args)
    logger = setup_logging(args)

    GENERATE_PROMPT, DEBUG_PROMPT, OPTIMIZE_PROMPT = setup_imports()

    print("Loading model and data...")
    model = get_model(args.model)
    data_processor = DataProcessor(DEBUG_PROMPT, OPTIMIZE_PROMPT)

    print("Loading datasets...")
    train_data = data_processor.load_dataset(args.dataset_path, args.train_file)
    dev_data = data_processor.load_dataset(args.dataset_path, args.dev_file)
    logger.info(
        f"Loaded {len(train_data)} training examples and {len(dev_data)} dev examples"
    )

    executor = Executor(model, args.temperature, args.seed)

    logger.info("\n=== Starting training loop ===\n")
    train_loop(
        args,
        model,
        train_data,
        dev_data,
        executor,
        data_processor,
        GENERATE_PROMPT,
    )
    logger.info("Training and evaluation on the dev data completed successfully")
    create_dev_summary(args)
    track_prompt_evolution(args)


def train_loop(
    args: argparse.Namespace,
    model: ModelBase,
    train_data: List[Dict],
    dev_data: List[Dict],
    executor: Executor,
    data_processor: DataProcessor,
    initial_prompt: str,
):
    """Training loop that implements the instruction improvement process."""
    # Initialize components
    run_metrics = RunMetrics(args.experiment_name, args.run_name)
    dev_evaluator = DevEvaluation(
        model, dev_data, executor, run_metrics, args.experiment_dir
    )
    total_batches = math.ceil(len(train_data) / args.batch_size)

    # Check if we're resuming from a checkpoint or starting fresh
    checkpoint = load_checkpoint(args.checkpoint_path)
    if checkpoint:
        start_epoch = checkpoint["last_completed_epoch"]
        start_batch = checkpoint["last_completed_batch"] + 1
        current_instruction = checkpoint["next_prompt"]
        logging.info(f"\nResuming from epoch {start_epoch}, batch {start_batch}")
    else:
        start_epoch = 1
        start_batch = 1
        current_instruction = initial_prompt
        logging.info("[Train] Starting new training run with prompt_0\n")

    # Process each epoch
    for epoch_num in range(start_epoch, args.epochs + 1):
        logging.info(f"Epoch {epoch_num}/{args.epochs}, starting epoch {epoch_num}")
        epoch_dir = os.path.join(args.training_dir, "epochs", f"epoch_{epoch_num}")
        batch_range_start = (
            start_batch if epoch_num == start_epoch else 1
        )  # For the resuming epoch, start from interrupted batch. For all other epochs, start from batch 1

        # Process each batch in current epoch
        for batch_num in range(batch_range_start, total_batches + 1):
            logging.info(
                f"Processing batch {batch_num}/{total_batches} with prompt_{batch_num-1}"
            )
            batch_dir = os.path.join(epoch_dir, f"batch_{batch_num}")
            os.makedirs(os.path.join(batch_dir, "individual_tasks"), exist_ok=True)

            # Extract current batch of data
            batch_start_idx = (batch_num - 1) * args.batch_size
            batch_end_idx = min(batch_num * args.batch_size, len(train_data))
            batch = train_data[batch_start_idx:batch_end_idx]

            # Process batch and get updated instruction
            logging.debug(f"Processing tasks {batch_start_idx + 1}-{batch_end_idx}")
            batch_results, batch_analysis, updated_instruction = (
                data_processor.process_batch(batch, model, args, current_instruction)
            )

            # Save individual task results
            for result, debug_result in zip(
                batch_results, batch_analysis["comparisons"]
            ):
                task_result = {
                    "task_id": result["task_id"],
                    "generated_code": result["generated_code"],
                    "reference_code": result["reference_code"],
                    "debug_comparison": debug_result,
                }
                task_path = os.path.join(
                    batch_dir, "individual_tasks", f"task_{result['task_id']}.json"
                )
                with open(task_path, "w") as f:
                    json.dump(task_result, f, indent=2)

            # Save combined batch results
            batch_record = {
                "instruction": {
                    "current": current_instruction,
                    "updated": updated_instruction,
                },
                "batch_results": [
                    {
                        "task_id": result["task_id"],
                        "generated_code": result["generated_code"],
                        "reference_code": result["reference_code"],
                        "debug_comparison": debug_result,
                    }
                    for result, debug_result in zip(
                        batch_results, batch_analysis["comparisons"]
                    )
                ],
            }
            with open(os.path.join(batch_dir, "batch_results.json"), "w") as f:
                json.dump(batch_record, f, indent=2)

            # Evaluate current instruction on dev set
            logging.info("\n[EVAL] Running dev evaluation")
            dev_evaluator.evaluate_instruction(
                current_instruction, epoch_num, batch_num
            )

            # Save checkpoint after successful batch processing
            save_checkpoint(
                args,
                epoch_num,
                batch_num,
                current_instruction,  # Prompt used for this batch
                updated_instruction,  # Prompt to use for next batch
            )

            # Update instruction for next batch
            current_instruction = updated_instruction

        # After completing an epoch, reset batch counter for next epoch
        start_batch = 1


if __name__ == "__main__":
    main()
