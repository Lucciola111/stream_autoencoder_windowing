
def evaluate_detection_times(actual_drift_points, detected_drift_points):

    """

    Parameters
    ----------
    actual_drift_points: Array with the actual drift points
    detected_drift_points: Array with the detected drift points by SAW

    Returns an array with the time until each actual drift is detected (if it was detected)
    -------

    """

    # Fill arrays with time until drift is detected (if detected)
    detection_times = []
    for idx in range(len(actual_drift_points)):
        detection_time = False
        for detected_drift in detected_drift_points:
            if actual_drift_points[idx] < detected_drift and not detection_time:
                next_actual_drift = actual_drift_points[idx + 1] if (idx + 1) < len(actual_drift_points) else False
                if detected_drift < next_actual_drift or not next_actual_drift:
                    detection_time = detected_drift - actual_drift_points[idx]
                    detection_times.append(detection_time)

    return detection_times
