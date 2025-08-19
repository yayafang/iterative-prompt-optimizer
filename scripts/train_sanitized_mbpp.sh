#!/bin/bash

# Go to the project root and activate virtual environment
cd "/c/Users/Hsin-Ya/Desktop/fresh"
source "./venv/Scripts/activate"

# Load OpenAI key
export $(cat .env | xargs)

export PYTHONPATH="/c/Users/Hsin-Ya/Desktop/fresh/code:$PYTHONPATH"

# Go to the code directory
cd "./code/src/train_sanitized_mbpp"

# Run the Python script
python main.py \
    --experiment_name "Exp4" \
    --run_name "Run1" \
    --root_dir "/c/Users/Hsin-Ya/Desktop/fresh/root/sanitized_mbpp" \
    --dataset_path "/c/Users/Hsin-Ya/Desktop/fresh/dataset/1_sanitized_mbpp" \
    --train_file "small_train.json" \
    --dev_file "dev.json" \
    --test_file "test.json" \
    --model "gpt4-turbo" \
    --seed "42" \
    --temperature "0.0" \
    --batch_size "1" \
    --epochs "1" 
