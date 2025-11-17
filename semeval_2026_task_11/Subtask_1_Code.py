from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
import evaluate

# -----------------------------
# 1. Load your CSV datasets
# -----------------------------
# Load training data from eng.csv
train_dataset = load_dataset("csv", data_files="/content/eng.csv")["train"]

# Load test data from eng_test.csv
test_dataset = load_dataset("csv", data_files="/content/eng_test.csv")["train"]

# Print dataset column names to help identify correct labels and text column
print("Available columns in training dataset:", train_dataset.column_names)
print("Available columns in test dataset:", test_dataset.column_names)

# The label columns in your dataset
label_cols = ["anger", "fear", "joy", "sadness", "surprise"]
num_labels = len(label_cols)

# -----------------------------
# 2. Tokenizer
# -----------------------------
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

train_ds = train_dataset.map(tokenize, batched=True)
test_ds = test_dataset.map(tokenize, batched=True)

# -----------------------------
# 3. Format labels
# -----------------------------
def format_labels(batch):
    labels_np = np.column_stack([batch[col] for col in label_cols])
    # Convert to float first to properly identify NaN values if any exist in the CSV
    labels_float = labels_np.astype(float)

    # Check for NaN values. If any are present, fill them with 0.
    # In multi-label classification, a missing label typically means the absence of that label.
    if np.isnan(labels_float).any():
        print("Warning: NaN values found in original labels. Replacing with 0.")
        labels_float = np.nan_to_num(labels_float, nan=0.0)

    # Convert to float32, which is expected by BCEWithLogitsLoss
    batch["labels"] = labels_float.astype(np.float32).tolist()
    return batch

train_ds = train_ds.map(format_labels, batched=True)
test_ds = test_ds.map(format_labels, batched=True)

# Remove unused columns
train_ds = train_ds.remove_columns(["id", "text"])
test_ds = test_ds.remove_columns(["id", "text"])

train_ds.set_format("torch")
test_ds.set_format("torch")

# -----------------------------
# 4. Load model
# -----------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    problem_type="multi_label_classification"
)

# -----------------------------
# 5. Metrics
# -----------------------------
f1_metric = evaluate.load("f1")
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)) # Convert logits to probabilities
    preds = (probs > 0.5).int().numpy().astype(np.int32) # Ensure predictions are int32

    # labels is a numpy array from the Trainer's evaluation_loop because set_format("torch")
    # converts the dataset elements to torch tensors, but the trainer's eval_loop yields numpy arrays.
    # After fixing format_labels, `labels` should already be clean floats. Convert them to int for metrics calculation
    references_for_metrics = labels.astype(np.int32) # Ensure labels are int32 for metrics

    # Flatten the predictions and references for the metrics
    # This is a common requirement for evaluate metrics in multi-label scenarios with 'micro' averaging
    preds_flat = preds.flatten()
    references_flat = references_for_metrics.flatten()

    f1 = f1_metric.compute(predictions=preds_flat, references=references_flat, average="micro")
    acc = accuracy_metric.compute(predictions=preds_flat, references=references_flat)

    return {
        "micro_f1": f1["f1"],
        "accuracy": acc["accuracy"]
    }

# -----------------------------
# 6. Training Arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    load_best_model_at_end=True,
    report_to="none"
)

# -----------------------------
# 7. Train the model
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# -----------------------------
# 8. Evaluate final accuracy
# -----------------------------
results = trainer.evaluate()
print(results)