import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from train_subtask_2 import MultitaskDeberta  # Import the model class

# -----------------------------
# 1. Configuration
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results_subtask_2")
TEST_FILE = os.path.join(BASE_DIR, "..", "test_data_subtask_2.json") # Assuming it's in the project root
OUTPUT_FILE = os.path.join(BASE_DIR, "predictions_subtask_2.json")
MAX_LENGTH = 256
MAX_PREMISES = 6

def get_latest_checkpoint(results_dir):
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    subdirs = [os.path.join(results_dir, d) for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d)) and d.startswith("checkpoint")]
    if not subdirs:
        raise ValueError(f"No checkpoints found in {results_dir}")
    latest_checkpoint = max(subdirs, key=os.path.getmtime)
    print(f"Using latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint

def predict():
    # 1. Load Model and Tokenizer
    checkpoint_path = get_latest_checkpoint(RESULTS_DIR)
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    config = AutoConfig.from_pretrained(checkpoint_path)
    
    # Ensure config has our custom attributes if they weren't saved correctly
    if not hasattr(config, 'num_premises'):
        config.num_premises = MAX_PREMISES
    
    model = MultitaskDeberta.from_pretrained(checkpoint_path, config=config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # 2. Load Data
    test_file_path = TEST_FILE
    print(f"Checking for test data at: {test_file_path}")
    
    if not os.path.exists(test_file_path):
        # Fallback for robustness
        alt_path = os.path.join(BASE_DIR, "train_data.json")
        if os.path.exists(alt_path):
             test_file_path = alt_path
             print(f"Using fallback test data at: {test_file_path}")
        else:
             raise FileNotFoundError(f"Could not find test file at {TEST_FILE} or {alt_path}")

    print(f"Loading test data from: {test_file_path}")
    with open(test_file_path, 'r') as f:
        data = json.load(f)
        
    predictions = []
    
    print("Running inference...")
    with torch.no_grad():
        for i, item in enumerate(data):
            text = item["syllogism"]
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            
            # Validity Prediction
            validity_logits = outputs.validity_logits
            validity_pred_idx = torch.argmax(validity_logits, dim=1).item()
            is_valid = bool(validity_pred_idx == 1)
            
            # Premise Prediction (Multi-label)
            premise_logits = outputs.premise_logits
            # Sigmoid to get probabilities
            premise_probs = torch.sigmoid(premise_logits)
            # Threshold at 0.5
            premise_preds = (premise_probs > 0.5).int().cpu().numpy()[0]
            
            # Convert binary vector to list of indices
            selected_premises = [idx for idx, val in enumerate(premise_preds) if val == 1]
            
            predictions.append({
                "id": item["id"],
                "validity": is_valid,
                "premises": selected_premises
            })
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(data)} items")
                
    # 3. Save Predictions
    print(f"Saving predictions to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(predictions, f, indent=4)
        
    return predictions

if __name__ == "__main__":
    predict()

