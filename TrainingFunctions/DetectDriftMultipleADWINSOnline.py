import numpy as np
from skmultiflow.drift_detection.adwin import ADWIN


def detect_drift_multiple_adwins_online(test_X, idx, adwins, agreement_threshold, reinitialize_adwin=False,
                                        model=False, test_y=False):

    """

    Parameters
    ----------
    test_X: Data stream with all test data
    idx: Index of outer loop
    adwins: Dictionary with one ADWIN change detector for each dimension
    agreement_threshold: Minimum of dimensions/ADWINs that need to indicate a drift at that instance
    reinitialize_adwin: Determines whether ADWIN should be initialized after detected drift (ADWIN-Xi)
    model: The model if algorithm should be validated with classifier
    test_y: The test labels if algorithm should be validated with classifier

    Returns the width of ADWIN and whether a drift point was detected
    -------

    """

    n_dimensions = test_X.shape[1]
    adwins_width = []
    drift_decision_fusion = False
    drift_point_dims = [False] * n_dimensions
    # Adding stream elements to ADWIN and verifying if drift occurred
    for dim in range(n_dimensions):
        adwins[dim].add_element(test_X[idx][dim])
        if adwins[dim].detected_change():
            print(f"Change in index {idx} in dimension {dim} for stream value {test_X[idx][dim]}")
            drift_point_dims[dim] = True

        adwins_width.append(adwins[dim].width)

    # If at least as many dimensions indicate a drift as required by the agreement rate
    if np.sum(drift_point_dims) >= agreement_threshold:
        drift_decision_fusion = True
        if model and test_y.any():
            # Identify smallest ADWIN width
            smallest = min(adwins_width)
            # Test-then-Train: Define ADWIN window as new train data stream
            window_train_X = test_X[(idx - smallest): idx]
            window_train_y = test_y[(idx - smallest): idx]
            model.fit(window_train_X, window_train_y)
        if reinitialize_adwin:
            for dim in range(n_dimensions):
                # Initialize ADWINs
                adwins[dim] = ADWIN(delta=0.002)

    return adwins, adwins_width, drift_decision_fusion, drift_point_dims
