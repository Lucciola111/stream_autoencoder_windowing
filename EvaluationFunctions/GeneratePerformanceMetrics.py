import pandas as pd
import numpy as np
from EvaluationFunctions.EvaluateWithLabels import evaluate_with_labels


def generate_performance_metrics(drift_decisions, drift_labels, acceptance_levels, simple_metrics, complex_metrics,
                                 index_name):
    # Fill array with indices of drift points
    detected_drift_points = list(np.where(drift_decisions)[0]) if index_name != 'PCA-CD' else drift_decisions
    # Fill array with indices of drift labels
    actual_drift_points = list(np.where(drift_labels)[0])

    # Generate performance metrics
    percentage_changes_detected, mean_time_until_detection, eval_TP, eval_FP, eval_FN, eval_precision, eval_recall, eval_f1score = evaluate_with_labels(
        actual_drift_points=actual_drift_points,
        detected_drift_points=detected_drift_points,
        acceptance_levels=acceptance_levels)

    data = {'% Changes detected': round(percentage_changes_detected, 2),
            'Mean time until detection (instances)': mean_time_until_detection}
    if simple_metrics:
        data = {**data, **eval_TP, **eval_FP, **eval_FN}
    if complex_metrics:
        data = {**data, **eval_precision, **eval_recall, **eval_f1score}

    performance_metrics = pd.DataFrame(data=data, index=[index_name])

    return performance_metrics
