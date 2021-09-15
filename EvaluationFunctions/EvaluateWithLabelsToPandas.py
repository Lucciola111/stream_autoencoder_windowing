import numpy as np
from EvaluationFunctions.EvaluateDetectionTimes import evaluate_detection_times
from EvaluationFunctions.EvaluatePerformanceMatrix import evaluate_performance_matrix


def evaluate_with_labels(actual_drift_points, detected_drift_points, acceptance_levels):
    """

    Parameters
    ----------
    actual_drift_points: Array with the actual drift points
    detected_drift_points: Array with the detected drift points by SAW
    acceptance_levels: Acceptance levels whether drift was detected in time

    Returns the performance metrics in a pandas data frame
    -------

    """

    # 0. Preparation: Calculate detection times
    detection_times = evaluate_detection_times(actual_drift_points=actual_drift_points,
                                               detected_drift_points=detected_drift_points)
    # 1. Calculate the percentage of drifts detected
    percentage_changes_detected = (len(detected_drift_points) / len(actual_drift_points)) * 100
    print('Percentage of drifts detected: {}%'.format(np.round(percentage_changes_detected, 2)))
    # 2. Mean time until detection
    mean_time_until_detection = sum(detection_times) / len(detection_times)
    print('Mean time until detection: {} instances'.format(np.round(mean_time_until_detection, 2)))
    # 3. Calculate TP, FP and FN for each given acceptance level
    true_positives, false_positives, false_negatives = {}, {}, {}
    precisions, recalls, f1scores = {}, {}, {}
    for acceptance_level in acceptance_levels:
        TP, FP, FN, precision, recall, f1score = evaluate_performance_matrix(
            actual_drift_points=actual_drift_points, detected_drift_points=detected_drift_points,
            detection_times=detection_times, acceptance_level=acceptance_level)
        true_positives[acceptance_level] = TP
        false_positives[acceptance_level] = FP
        false_negatives[acceptance_level] = FN
        precisions[acceptance_level] = precision
        recalls[acceptance_level] = recall
        f1scores[acceptance_level] = f1score

    return percentage_changes_detected, mean_time_until_detection, true_positives, false_positives, false_negatives, precisions, recalls, f1scores
