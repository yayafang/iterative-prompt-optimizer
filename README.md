
  # AI-Powered Code Generation Improvement Framework

  This project demonstrates an iterative prompt optimization system that uses Large Language Models (LLMs) to
  generate Python code and automatically improves the generation prompts based on performance feedback.

  ## Overview

  The system works by:
  1. **Code Generation**: Using GPT models to generate Python functions from natural language prompts
  2. **Performance Analysis**: Comparing generated code with reference implementations through test execution
  3. **Prompt Evolution**: Automatically refining generation prompts based on failure analysis
  4. **Iterative Learning**: Continuously improving code generation quality across training batches

  ## Key Features

  - **Multi-model Support**: Compatible with any models
  - **Automated Prompt Optimization**: Uses debugger and optimizer prompts to enhance generation instructions
  - **Comprehensive Evaluation**: Tracks performance on development and test sets
  - **Execution Safety**: Sandboxed code execution with timeout protection
  - **Detailed Logging**: Complete experiment tracking and prompt evolution history

  ## Project Structure

  ├── main_only_train_evalplus.py    # Main training script
  ├── models.py                      # OpenAI model wrappers
  ├── data_processor.py              # Core batch processing logic
  ├── executor.py                    # Safe code execution engine
  ├── dev_eval.py                   # Development set evaluation
  ├── metrics.py                    # Performance tracking
  ├── prompts/
  │   ├── generator.py              # Initial generation prompt
  │   ├── debugger.py               # Code analysis prompt
  │   └── optimizer.py              # Prompt improvement prompt
  ├── dataset/
  │   ├── small_train.json          # Training examples
  │   ├── small_dev.json            # Development examples
  │   └── small_test.json           # Test examples
  ├── requirements.txt              # Dependencies
  └── README.md                     # This file

  ## Installation

  1. **Clone the repository**
  ```bash
  git clone <your-repo-url>
  cd <repo-name>

  2. Create virtual environment
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate

  3. Install dependencies
  pip install -r requirements.txt

  4. Set up OpenAI API key
  export OPENAI_API_KEY="your-api-key-here"

  Usage

  Basic Training Run

  python main_only_train_evalplus.py \
      --experiment_name "demo_experiment" \
      --run_name "run_1" \
      --root_dir "./results" \
      --dataset_path "./dataset" \
      --train_file "small_train.json" \
      --dev_file "small_dev.json" \
      --model "gpt-4o-mini" \
      --batch_size 5 \
      --epochs 1 \
      --eval_interval 2

  Training-Only Mode (Skip Evaluations)

  python main_only_train_evalplus.py \
      --experiment_name "training_only" \
      --run_name "run_1" \
      --root_dir "./results" \
      --dataset_path "./dataset" \
      --train_file "small_train.json" \
      --model "gpt-4o-mini" \
      --batch_size 5 \
      --epochs 1 \
      --skip_evaluations

  Command Line Arguments

  Required Arguments

  - --experiment_name: Name for your experiment
  - --run_name: Specific run identifier
  - --root_dir: Directory to save results
  - --dataset_path: Path to dataset folder
  - --train_file: Training data filename
  - --model: Model to use (gpt4-turbo, gpt-4o-mini, gpt-3.5-turbo)

  Optional Arguments

  - --dev_file: Development data filename
  - --test_file: Test data filename
  - --batch_size: Number of examples per batch (default: 10)
  - --epochs: Number of training epochs (default: 1)
  - --eval_interval: Batches between evaluations (default: 5)
  - --temperature: Model temperature (default: 0.0)
  - --seed: Random seed (default: 42)
  - --skip_evaluations: Skip all evaluations, training only
  - --base_test: Use base test evaluation
  - --plus_test: Use plus test evaluation
  - --require_both: Require both base and plus tests to pass

  Data Format

  Training data should be JSON files with the following structure:

  [
    {
      "task_id": 602,
      "prompt": "Write a python function to find the first repeated character in a given string.",
      "code": "def first_repeated_char(str1):\n  for index,c in enumerate(str1):\n    if str1[:index+1].count(c) >      
  1:\n      return c",
      "test": [
        "assert first_repeated_char(\"abcabc\") == \"a\"",
        "assert first_repeated_char(\"abc\") == None"
      ]
    }
  ]

  Output Structure

  After running an experiment, you'll find:

  results/
  └── demo_experiment/
      ├── logs/
      │   └── experiment.log           # Detailed execution logs
      ├── training/
      │   ├── epochs/
      │   │   └── epoch_1/
      │   │       └── batch_*/
      │   │           ├── batch_results.json
      │   │           └── individual_tasks/
      │   ├── checkpoints/
      │   │   └── latest.json          # Resume checkpoint
      │   ├── prompt_evolution.json    # Complete prompt history
      │   └── final_prompt.txt        # Best performing prompt
      └── dev_evaluation/             # Development set results
          └── epoch_*/
              └── batch_*/
                  ├── dev_evaluation.json
                  └── individual_tasks/

  Key Components

  Models (models.py)

  - Unified interface for OpenAI GPT models
  - Automatic retry logic with exponential backoff
  - Consistent message formatting

  Data Processor (data_processor.py)

  - Batch processing of training examples
  - Code generation from prompts
  - Debug analysis and prompt optimization

  Executor (executor.py)

  - Safe code execution in isolated environment
  - Test case validation
  - Timeout protection and error handling

  Evaluation (dev_eval.py)

  - Performance assessment on development set
  - Individual task tracking
  - Comprehensive result logging

  Research Context

  This framework was developed as part of research into automated prompt engineering for code generation. The
  system demonstrates how iterative feedback can improve LLM performance on programming tasks without requiring
  model retraining.