import json
import math
from typing import List, Dict, Any, Tuple, Optional

# Load prediction file
PREDICTIONS_FILE = "predictions.json"
TRAIN_DATA_FILE = "semeval_2026_task_11/train_data/task 1/train_data.json"

# Import functions from the original evaluation script
# We are copying them here to ensure independence and ease of running
def calculate_accuracy(
    ground_truth_list: List[Dict[str, Any]],
    predictions_list: List[Dict[str, Any]],
    metric_name: str,
    prediction_key: str,
    plausibility_filter: Optional[bool] = None
) -> Tuple[float, int, int]:
    gt_map = {item['id']: item for item in ground_truth_list}
    correct_predictions = 0
    total_predictions = 0

    for pred_item in predictions_list:
        item_id = pred_item['id']
        if item_id in gt_map:
            gt_item = gt_map[item_id]
            gt_plausibility = gt_item.get('plausibility')
            if plausibility_filter is not None and gt_plausibility != plausibility_filter:
                continue 
            
            if metric_name in gt_item and prediction_key in pred_item:
                true_label = gt_item[metric_name]
                predicted_label = pred_item[prediction_key]

                if isinstance(true_label, bool) and isinstance(predicted_label, bool):
                    total_predictions += 1
                    if true_label == predicted_label:
                        correct_predictions += 1

    if total_predictions == 0:
        return 0.0, 0, 0

    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy, correct_predictions, total_predictions


def calculate_subgroup_accuracy(
    gt_map: Dict[str, Any],
    predictions_list: List[Dict[str, Any]],
    gt_validity: bool,
    gt_plausibility: bool
) -> Tuple[float, int, int]:
    correct_predictions = 0
    total_predictions = 0
    
    for pred_item in predictions_list:
        item_id = pred_item['id']
        if item_id in gt_map:
            gt_item = gt_map[item_id]
            if gt_item.get('validity') == gt_validity and gt_item.get('plausibility') == gt_plausibility:
                if 'validity' in gt_item and 'validity' in pred_item:
                    true_label = gt_item['validity']
                    predicted_label = pred_item['validity']
                    if isinstance(true_label, bool) and isinstance(predicted_label, bool):
                        total_predictions += 1
                        if true_label == predicted_label:
                            correct_predictions += 1

    if total_predictions == 0:
        return 0.0, 0, 0

    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy, correct_predictions, total_predictions


def calculate_content_effect_bias(accuracies: Dict[str, float]) -> Dict[str, float]:
    acc_plausible_valid = accuracies.get('acc_plausible_valid', 0.0)
    acc_implausible_valid = accuracies.get('acc_implausible_valid', 0.0)
    acc_plausible_invalid = accuracies.get('acc_plausible_invalid', 0.0)
    acc_implausible_invalid = accuracies.get('acc_implausible_invalid', 0.0)

    intra_valid_diff = abs(acc_plausible_valid - acc_implausible_valid)
    intra_invalid_diff = abs(acc_plausible_invalid - acc_implausible_invalid)
    content_effect_intra_validity_label = (intra_valid_diff + intra_invalid_diff) / 2.0

    inter_plausible_diff = abs(acc_plausible_valid - acc_plausible_invalid)
    inter_implausible_diff = abs(acc_implausible_valid - acc_implausible_invalid) # NOTE: This seems to be a copy-paste error in original, likely meant acc_implausible_valid - acc_implausible_invalid? No wait, original script says: abs(acc_implausible_valid - acc_implausible_invalid) at line 169.
    # Let's re-read the original script carefully.
    # Line 168: inter_plausible_diff = abs(acc_plausible_valid - acc_plausible_invalid)
    # Line 169: inter_implausible_diff = abs(acc_implausible_valid - acc_implausible_invalid)
    # Yes.
    inter_implausible_diff = abs(acc_implausible_valid - acc_implausible_invalid)
    
    content_effect_inter_validity_label = (inter_plausible_diff + inter_implausible_diff) / 2.0
    tot_content_effect = (content_effect_intra_validity_label + content_effect_inter_validity_label) / 2.0
    
    return {
        'content_effect_intra_validity_label': content_effect_intra_validity_label,
        'content_effect_inter_validity_label': content_effect_inter_validity_label,
        'tot_content_effect': tot_content_effect
    }

def calculate_smooth_combined_metric(overall_accuracy: float, total_content_effect: float) -> float:
    if total_content_effect < 0:
        return 0.0
    log_penalty = math.log(1 + total_content_effect)
    combined_smooth_score = overall_accuracy / (1 + log_penalty)
    return combined_smooth_score

def main():
    print(f"Loading predictions from {PREDICTIONS_FILE}...")
    with open(PREDICTIONS_FILE, 'r') as f:
        predictions = json.load(f)
        
    print(f"Loading ground truth from {TRAIN_DATA_FILE}...")
    with open(TRAIN_DATA_FILE, 'r') as f:
        all_ground_truth = json.load(f)

    # Create a map of ground truth
    gt_map_all = {item['id']: item for item in all_ground_truth}
    
    # Filter ground truth to only include IDs present in predictions (Validation Set)
    pred_ids = set(item['id'] for item in predictions)
    ground_truth = [gt_map_all[pid] for pid in pred_ids if pid in gt_map_all]
    gt_map = {item['id']: item for item in ground_truth}
    
    print(f"Evaluated Samples: {len(predictions)}")

    # --- ANALYSIS ---
    model_name = "RoBERTa-Base (Validation)"
    
    # 1. Overall Accuracy
    overall_acc, overall_correct, overall_total = calculate_accuracy(
        ground_truth, predictions, 'validity', 'validity', None
    )
    print(f"\nOverall Accuracy: {overall_acc:.2f}% ({overall_correct}/{overall_total})")

    # 2. Plausible vs Implausible Accuracy
    plausible_acc, _, _ = calculate_accuracy(ground_truth, predictions, 'validity', 'validity', True)
    implausible_acc, _, _ = calculate_accuracy(ground_truth, predictions, 'validity', 'validity', False)
    print(f"Plausible Accuracy: {plausible_acc:.2f}%")
    print(f"Implausible Accuracy: {implausible_acc:.2f}%")

    # 3. Content Bias Metrics
    acc_plausible_valid, _, _ = calculate_subgroup_accuracy(gt_map, predictions, True, True)
    acc_implausible_valid, _, _ = calculate_subgroup_accuracy(gt_map, predictions, True, False)
    acc_plausible_invalid, _, _ = calculate_subgroup_accuracy(gt_map, predictions, False, True)
    acc_implausible_invalid, _, _ = calculate_subgroup_accuracy(gt_map, predictions, False, False)

    conditional_accuracies = {
        'acc_plausible_valid': acc_plausible_valid,
        'acc_implausible_valid': acc_implausible_valid,
        'acc_plausible_invalid': acc_plausible_invalid,
        'acc_implausible_invalid': acc_implausible_invalid
    }
    
    print("\nConditional Accuracies:")
    print(f"  Valid & Plausible: {acc_plausible_valid:.2f}%")
    print(f"  Valid & Implausible: {acc_implausible_valid:.2f}%")
    print(f"  Invalid & Plausible: {acc_plausible_invalid:.2f}%")
    print(f"  Invalid & Implausible: {acc_implausible_invalid:.2f}%")

    bias_metrics = calculate_content_effect_bias(conditional_accuracies)
    tot_content_effect = bias_metrics['tot_content_effect']
    
    print(f"\nTotal Content Effect (Bias): {tot_content_effect:.2f}%")

    # 4. Combined Score
    combined_score = calculate_smooth_combined_metric(overall_acc, tot_content_effect)
    
    print("\n" + "=" * 50)
    print(f"FINAL COMBINED SCORE: {combined_score:.2f}")
    print("=" * 50)

if __name__ == "__main__":
    main()


