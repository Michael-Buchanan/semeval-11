# Run !pip install transformers datasets evaluate accelerate torch before running the code
# To run this code, "train_data.json" AND "Pilot_Data.json" need to be in the same file as the code

import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
train_file = "train_data.json"
pilot_file = "test_data_subtask_1.json"
model_name = "roberta-large-mnli"
max_length = 256
batch_size = 8 
num_epochs = 10 
learning_rate = 1e-5 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device} | Model: {model_name}")
train_raw = load_dataset("json", data_files=train_file)
pilot_raw = load_dataset("json", data_files=pilot_file)
full_train_ds = train_raw["train"] if "train" in train_raw else train_raw
pilot_ds = pilot_raw["train"] if "train" in pilot_raw else pilot_raw
split_ds = full_train_ds.train_test_split(test_size=0.1, seed=42)
train_ds = split_ds["train"]
eval_ds = split_ds["test"]
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
def preprocess_train(batch):
    inputs = [f"Determine validity: {x}" for x in batch["syllogism"]]
    enc = tokenizer(inputs, truncation=True, max_length=max_length)
    enc["labels"] = [int(v) for v in batch["validity"]]
    return enc

def preprocess_pilot(batch):
    inputs = [f"Determine validity: {x}" for x in batch["syllogism"]]
    enc = tokenizer(inputs, truncation=True, max_length=max_length)
    enc["id"] = batch["id"]
    return enc

train_ds = train_ds.map(preprocess_train, batched=True, remove_columns=train_ds.column_names)
eval_ds = eval_ds.map(preprocess_train, batched=True, remove_columns=eval_ds.column_names)
pilot_ds = pilot_ds.map(preprocess_pilot, batched=True, remove_columns=pilot_ds.column_names)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    return {"accuracy": acc, "f1": f1}
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    ignore_mismatched_sizes=True
).to(device)

collator = DataCollatorWithPadding(tokenizer)
args = TrainingArguments(
    output_dir="./model_output",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    learning_rate=learning_rate,
    weight_decay=0.01,
    warmup_ratio=0.1,

    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
)

trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds, data_collator=collator, tokenizer=tokenizer, compute_metrics=compute_metrics)
trainer.train()
print("Predicting...")
pred_output = trainer.predict(pilot_ds)
logits = pred_output.predictions
probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
preds = np.argmax(probs, axis=-1)
ids = np.array(pilot_ds["id"])

pred_items = []
for i, example_id in enumerate(ids):
    pred_items.append({"id": str(example_id), "validity": bool(int(preds[i]))})

with open("predictions.json", "w") as f:
    json.dump(pred_items, f, indent=2)

print("predictions.json created")
