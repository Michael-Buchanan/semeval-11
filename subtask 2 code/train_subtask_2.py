import json
import numpy as np
import torch
import torch.nn as nn
import os
from dataclasses import dataclass
from typing import Optional, Tuple
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoConfig,
    DebertaV2PreTrainedModel, 
    DebertaV2Model,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding,
    modeling_outputs
)
import evaluate
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. Configuration
# -----------------------------
MODEL_NAME = "microsoft/deberta-v3-base"

# Robust path handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(BASE_DIR, "train_data.json")

OUTPUT_DIR = "./results_subtask_2"
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_PREMISES = 6 

# -----------------------------
# 2. Custom Multitask Model
# -----------------------------
@dataclass
class MultitaskOutput(modeling_outputs.ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    validity_logits: torch.FloatTensor = None
    premise_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class MultitaskDeberta(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels # Validity (2)
        self.num_premises = config.num_premises # Premises (e.g. 6)
        
        self.deberta = DebertaV2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Head 1: Validity (Binary)
        self.validity_classifier = nn.Linear(config.hidden_size, self.num_labels)
        
        # Head 2: Premise Selection (Multi-label)
        self.premise_classifier = nn.Linear(config.hidden_size, self.num_premises)
        
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,        # Validity Labels
        premise_labels=None, # Premise Labels (Multi-hot)
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = encoder_layer[:, 0, :] # [CLS] token
        pooled_output = self.dropout(pooled_output)

        # Heads
        validity_logits = self.validity_classifier(pooled_output)
        premise_logits = self.premise_classifier(pooled_output)

        loss = None
        if labels is not None:
            # 1. Validity Loss (CrossEntropy)
            loss_fct_val = nn.CrossEntropyLoss()
            loss_val = loss_fct_val(validity_logits.view(-1, self.num_labels), labels.view(-1))
            loss = loss_val

            # 2. Premise Loss (BCEWithLogits) - Only if premise_labels provided
            if premise_labels is not None:
                loss_fct_prem = nn.BCEWithLogitsLoss()
                # Ensure premise_labels is float for BCE
                loss_prem = loss_fct_prem(premise_logits, premise_labels.float())
                loss += loss_prem

        return MultitaskOutput(
            loss=loss,
            validity_logits=validity_logits,
            premise_logits=premise_logits,
            hidden_states=outputs.last_hidden_state,
            attentions=outputs.attentions,
        )

# -----------------------------
# 3. Data Loading & Preprocessing
# -----------------------------
def load_data(file_path):
    print(f"Attempting to load data from: {file_path}")
    if not os.path.exists(file_path):
             raise FileNotFoundError(f"File not found: {file_path}")
        
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    formatted_data = {
        "id": [],
        "text": [],
        "label": [],
        "premise_labels": [] 
    }
    
    for item in data:
        formatted_data["id"].append(item["id"])
        formatted_data["text"].append(item["syllogism"])
        formatted_data["label"].append(1 if item["validity"] else 0)
        
        # MOCK PREMISE LABELS
        mock_premise_vector = [0.0] * MAX_PREMISES
        if item["validity"]:
            mock_premise_vector[0] = 1.0
            mock_premise_vector[1] = 1.0
        
        formatted_data["premise_labels"].append(mock_premise_vector)
        
    return Dataset.from_dict(formatted_data)

print(f"Loading data from {TRAIN_FILE}...")
full_dataset = load_data(TRAIN_FILE)
dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_split["train"]
eval_dataset = dataset_split["test"]

# -----------------------------
# 4. Tokenization
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

# CRITICAL FIX: Remove non-tensor columns (id, text)
tokenized_train = tokenized_train.remove_columns(["id", "text"])
tokenized_eval = tokenized_eval.remove_columns(["id", "text"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# -----------------------------
# 5. Model Initialization
# -----------------------------
config = AutoConfig.from_pretrained(MODEL_NAME)
config.num_labels = 2
config.num_premises = MAX_PREMISES

model = MultitaskDeberta.from_pretrained(MODEL_NAME, config=config)

# -----------------------------
# 6. Metrics
# -----------------------------
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    
    # predictions is a tuple: (validity_logits, premise_logits, hidden_states)
    validity_logits = predictions[0] 
    validity_preds = np.argmax(validity_logits, axis=1)
    
    # labels is a tuple: (validity_labels, premise_labels)
    if isinstance(labels, tuple):
        validity_labels = labels[0]
    else:
        validity_labels = labels
        
    acc = accuracy_metric.compute(predictions=validity_preds, references=validity_labels)
    
    return {
        "validity_accuracy": acc["accuracy"],
    }

# -----------------------------
# 7. Training
# -----------------------------
class MultitaskTrainer(Trainer):
    # Override compute_loss to avoid subscriptable error with custom output
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
    remove_unused_columns=False 
)

trainer = MultitaskTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting Multitask Training...")
trainer.train()
