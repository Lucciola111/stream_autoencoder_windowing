import numpy as np


def create_results_values(proxy_evaluation, drift_labels_known, drift_decisions, acc_vector, drift_labels):
    """

    Parameters
    ----------
    proxy_evaluation: Boolean whether proxy evaluation was used
    drift_labels_known: Boolean whether change detection evaluation is used
    drift_decisions: Array with booleans for decisions about drift
    acc_vector: Array with booleans for (in)correct classification of data instance
    drift_labels: Array with booleans of drift labels

    Returns the results data, the headers for results and the format of the results
    -------

    """

    # Save values for: drift_decisions, acc_vector, drift_labels
    if proxy_evaluation and drift_labels_known:
        result_data = np.transpose([drift_decisions, acc_vector, drift_labels])
        result_header = "driftDecisions, accVector, driftLabels"
        result_format = ('%i', '%i', '%i')
    elif proxy_evaluation:
        result_data = np.transpose([drift_decisions, acc_vector])
        result_header = "driftDecisions, accVector"
        result_format = ('%i', '%i')
    elif drift_labels_known:
        result_data = np.transpose([drift_decisions, drift_labels])
        result_header = "driftDecisions, driftLabels"
        result_format = ('%i', '%i')
    else:
        result_data = np.transpose([drift_decisions])
        result_header = "driftDecisions"
        result_format = '%i'

    return result_data, result_header, result_format
