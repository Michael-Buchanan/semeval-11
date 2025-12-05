import json
import os
import math
from typing import Dict, List, Tuple, Optional, Any

from llm_to_symbolic import LLMClient
from symbolic_solver import is_valid_syllogism

# =========================
# METRIC FUNCTIONS
# =========================

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


def analyze_model_performance(
    model_name: str,
    ground_truth: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    gt_map: Dict[str, Any]
) -> Dict[str, Any]:

    print("\n" + "#" * 70)
    print(f"--- RESULTS FOR {model_name} ---")
    print("#" * 70)

    common_args = {
        'ground_truth_list': ground_truth,
        'predictions_list': predictions,
        'metric_name': 'validity',
        'prediction_key': 'validity'
    }

    overall_acc, overall_correct, overall_total = calculate_accuracy(
        **common_args,
        plausibility_filter=None
    )
    print(f"Overall Data Count: {overall_total}")
    print("-" * 50)
    print(f"[Group: ALL] Validity Accuracy: {overall_acc:.2f}% ({overall_correct} / {overall_total})")

    plausible_acc, plausible_correct, plausible_total = calculate_accuracy(
        **common_args,
        plausibility_filter=True
    )
    print(f"\n[Group: PLAUSIBLE (Plausibility: True)] Validity Accuracy: {plausible_acc:.2f}% ({plausible_correct} / {plausible_total})")

    implausible_acc, implausible_correct, implausible_total = calculate_accuracy(
        **common_args,
        plausibility_filter=False
    )
    print(f"[Group: IMPLAUSIBLE (Plausibility: False)] Validity Accuracy: {implausible_acc:.2f}% ({implausible_correct} / {implausible_total})")

    print("\n" + "=" * 50)
    print("--- Content Effect Bias Metrics ---")

    acc_plausible_valid, _, total_pv = calculate_subgroup_accuracy(gt_map, predictions, gt_validity=True, gt_plausibility=True)
    acc_implausible_valid, _, total_iv = calculate_subgroup_accuracy(gt_map, predictions, gt_validity=True, gt_plausibility=False)
    acc_plausible_invalid, _, total_pi = calculate_subgroup_accuracy(gt_map, predictions, gt_validity=False, gt_plausibility=True)
    acc_implausible_invalid, _, total_ii = calculate_subgroup_accuracy(gt_map, predictions, gt_validity=False, gt_plausibility=False)

    conditional_accuracies = {
        'acc_plausible_valid': acc_plausible_valid,
        'acc_implausible_valid': acc_implausible_valid,
        'acc_plausible_invalid': acc_plausible_invalid,
        'acc_implausible_invalid': acc_implausible_invalid
    }

    print(f"\nConditional Accuracies (Validity Prediction):")
    print(f"  Valid & Plausible (V/P) (N={total_pv}): {acc_plausible_valid:.2f}%")
    print(f"  Valid & Implausible (V/I) (N={total_iv}): {acc_implausible_valid:.2f}%")
    print(f"  Invalid & Plausible (IV/P) (N={total_pi}): {acc_plausible_invalid:.2f}%")
    print(f"  Invalid & Implausible (IV/I) (N={total_ii}): {acc_implausible_invalid:.2f}%")

    bias_metrics = calculate_content_effect_bias(conditional_accuracies)
    tot_content_effect = bias_metrics['tot_content_effect']

    print("\nCalculated Content Effect Metrics (Bias towards Plausibility):")
    print(f"  Intra-Validity Bias (acc_intra): {bias_metrics['content_effect_intra_validity_label']:.2f}%")
    print(f"  Inter-Validity Bias (acc_inter): {bias_metrics['content_effect_inter_validity_label']:.2f}%")
    print(f"  Total Content Effect (acc_tot): {tot_content_effect:.2f}%")

    combined_smooth_score = calculate_smooth_combined_metric(overall_acc, tot_content_effect)

    print("\n" + "=" * 50)
    print("--- Combined Performance Scores ---")
    print(f"\n[2] Log-Smoothed Score (Accuracy / (1 + ln(1 + Bias))):")
    print(f"    Score: {combined_smooth_score:.2f}")
    print("=" * 50)

    

    return {
        'name': model_name,
        'overall_acc': overall_acc,
        'tot_content_effect': tot_content_effect,
        'combined_smooth_score': combined_smooth_score
    }


# =========================
# PATH CONSTANTS
# =========================

PILOT_DATA_PATH = "pilot data/syllogistic_reasoning_binary_pilot_en.json"
LOG_PATH = "pilot_results.jsonl"


# =========================
# PILOT EVAL (OPTION 1)
# =========================

def run_pilot_evaluation():
    # 1. Load pilot ground truth
    with open(PILOT_DATA_PATH, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    gt_map = {item["id"]: item for item in ground_truth}
    print(f"Loaded {len(ground_truth)} pilot examples from {PILOT_DATA_PATH}")

    # 2. Instantiate LLM client
    llm = LLMClient(model="gpt-5.1")  # adjust as needed
    print("Initialized LLM client.")

    predictions: List[Dict[str, Any]] = []

    # 3. Open log file
    print(f"Processing examples and writing detailed log to {LOG_PATH}...")
    with open(LOG_PATH, "w", encoding="utf-8") as log_f:
        for idx, item in enumerate(ground_truth):
            sid = item["id"]
            text = item["syllogism"]
            gold_validity = item["validity"]
            gold_plaus = item["plausibility"]

            try:
                structured = llm.parse_syllogism(text)
                predicted_validity = is_valid_syllogism(structured)
                predictions.append({"id": sid, "validity": predicted_validity})

                log_entry = {
                    "id": sid,
                    "syllogism": text,
                    "gold_validity": gold_validity,
                    "gold_plausibility": gold_plaus,
                    "predicted_validity": predicted_validity,
                    "correct": predicted_validity == gold_validity,
                    "structured": structured,
                }
                if idx % 5 == 0:
                    print(log_entry)

            except Exception as e:
                log_entry = {
                    "id": sid,
                    "syllogism": text,
                    "gold_validity": gold_validity,
                    "gold_plausibility": gold_plaus,
                    "predicted_validity": None,
                    "correct": False,
                    "error": str(e),
                }
                # Note: we don't append a prediction for this id,
                # so it won't be counted in metrics.

            log_f.write(json.dumps(log_entry) + "\n")

            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1} items...")

    # 4. Analyze performance
    results = analyze_model_performance(
        model_name="NeuroSymbolic-GPT5",
        ground_truth=ground_truth,
        predictions=predictions,
        gt_map=gt_map,
    )

    print("\n\nDetailed per-example log written to:", os.path.abspath(LOG_PATH))
    print("Summary metrics:", results)


# =========================
# LOG-BASED METRICS (OPTION 2)
# =========================

def analyze_log_file(
    log_path: str,
    model_name: str = "NeuroSymbolic-FromLog"
) -> Dict[str, Any]:
    """
    Load a JSONL results log, rebuild ground truth + predictions,
    and run the full metric pipeline.
    """
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return {}

    ground_truth_map: Dict[str, Dict[str, Any]] = {}
    predictions: List[Dict[str, Any]] = []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            sid = rec.get("id")
            if sid is None:
                continue

            gt_v = rec.get("gold_validity")
            gt_p = rec.get("gold_plausibility")

            if sid not in ground_truth_map and isinstance(gt_v, bool) and isinstance(gt_p, bool):
                ground_truth_map[sid] = {
                    "id": sid,
                    "validity": gt_v,
                    "plausibility": gt_p,
                }

            pred_v = rec.get("predicted_validity")
            if isinstance(pred_v, bool):
                predictions.append({"id": sid, "validity": pred_v})

    ground_truth = list(ground_truth_map.values())
    gt_map = {item["id"]: item for item in ground_truth}

    if not predictions:
        print("No valid predictions found in log file.")
        return {}

    print(f"Loaded {len(ground_truth)} ground-truth items and {len(predictions)} predictions from {log_path}")
    results = analyze_model_performance(
        model_name=model_name,
        ground_truth=ground_truth,
        predictions=predictions,
        gt_map=gt_map,
    )
    print("Summary metrics:", results)
    return results
