import json
import numpy as np
import torch
import glob
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)

# -----------------------------
# 1. Configuration
# -----------------------------
BASE_OUTPUT_DIR = "./results_subtask_1"
TEST_FILE = "test_data_subtask_1.json"
OUTPUT_FILE = "predictions.json"
MAX_LENGTH = 256

# -----------------------------
# 2. Find Best/Latest Model
# -----------------------------
# Find the checkpoint with the highest step number
checkpoints = glob.glob(f"{BASE_OUTPUT_DIR}/checkpoint-*")
if not checkpoints:
    raise FileNotFoundError(f"No checkpoints found in {BASE_OUTPUT_DIR}")

# Sort by step number
latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
MODEL_PATH = latest_checkpoint
print(f"Loading model from: {MODEL_PATH}")

# -----------------------------
# 3. Data Loading
# -----------------------------
print(f"Loading test data from {TEST_FILE}...")
with open(TEST_FILE, 'r') as f:
    test_data = json.load(f)

formatted_data = {
    "id": [],
    "text": []
}

for item in test_data:
    formatted_data["id"].append(item["id"])
    formatted_data["text"].append(item["syllogism"])

test_dataset = Dataset.from_dict(formatted_data)
print(f"Loaded {len(test_dataset)} test samples.")

# -----------------------------
# 4. Model & Tokenizer Loading
# -----------------------------
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback: Load config from base name but weights from checkpoint
    print("Attempting fallback loading...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# -----------------------------
# 5. Prediction
# -----------------------------
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)

tokenized_test = test_dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./tmp_prediction",
    per_device_eval_batch_size=32,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer
)

print("Running predictions...")
predictions_output = trainer.predict(tokenized_test)
logits = predictions_output.predictions
preds = np.argmax(logits, axis=1)

# -----------------------------
# 6. Save Results
# -----------------------------
output_json = []
ids = test_dataset["id"]

for idx, pred_label in enumerate(preds):
    output_json.append({
        "id": ids[idx],
        "validity": bool(pred_label == 1)
    })

with open(OUTPUT_FILE, "w") as f:
    json.dump(output_json, f, indent=4)

print(f"Predictions saved to {OUTPUT_FILE}")
print(f"Total predictions: {len(output_json)}")
