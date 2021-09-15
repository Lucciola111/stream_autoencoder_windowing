
def evaluate_performance_matrix(actual_drift_points, detected_drift_points, detection_times, acceptance_level):
    """

    Parameters
    ----------
    actual_drift_points: Array with the actual drift points
    detected_drift_points: Array with the detected drift points by SAW
    detection_times: Array with the time until each actual drift is detected (if it was detected)
    acceptance_level: Acceptance level whether drift was detected in time

    Returns TPR, FPR, FNR, Precision, Recall, F1 Score for an acceptance level
    -------

    """

    def safe_div(x, y):
        if y == 0:
            return 0
        return x / y

    print('Detection within {} instances accepted:'.format(acceptance_level))
    # Calculate True Positives
    TP = sum(t <= acceptance_level for t in detection_times)
    print('True Positives Rate after {} instances: {}'.format(acceptance_level, safe_div(TP, len(actual_drift_points))))
    # Calculate False Positives
    FP = len(detected_drift_points) - TP
    print('False Positive Rate (detection within {} instances): {}'.format(acceptance_level, safe_div(FP, len(actual_drift_points))))
    # Calculate False Negatives
    FN = len(actual_drift_points) - TP
    print('False Negative Rate (detection within {} instances): {}'.format(acceptance_level, safe_div(FN, len(actual_drift_points))))
    # Calculate Precision
    precision = safe_div(TP, (TP+FP))
    print('Precision (detection within {} instances): {}'.format(acceptance_level, precision))
    # Calculate Recall
    recall = safe_div(TP, (TP+FN))
    print('Recall (detection within {} instances): {}'.format(acceptance_level, recall))
    # Calculate F1-Score
    f1score = safe_div(2 * (precision*recall), (precision+recall))
    # f1score = (2 * (precision*recall) / (precision+recall)) if (precision+recall) > 0 else float(0.0)
    print('F1 Score (detection within {} instances): {}'.format(acceptance_level, f1score))

    return TP, FP, FN, precision, recall, f1score
