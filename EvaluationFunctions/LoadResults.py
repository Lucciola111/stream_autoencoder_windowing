import numpy as np
import os


def load_results(file_name, result_folder):
    """

    Parameters
    ----------
    file_name: Name of the file
    result_folder: Folder of the file

    Returns the drift decisions and the drift labels
    -------

    """

    evaluation_results = {}

    proxy_and_change_detection_evaluation = os.path.exists(
        '../Files_Results/' + str(result_folder) + '/ChangeDetectionEvaluation_and_ProxyEvaluation/' + str(
            file_name) + '.csv')

    change_detection_evaluation = os.path.exists(
        '../Files_Results/' + str(result_folder) + '/ChangeDetectionEvaluation/' + str(
            file_name) + '.csv')

    proxy_evaluation = os.path.exists(
        '../Files_Results/' + str(result_folder) + '/ProxyEvaluation/' + str(
            file_name) + '.csv')

    if proxy_and_change_detection_evaluation:
        path = '../Files_Results/' + str(result_folder) + '/ChangeDetectionEvaluation_and_ProxyEvaluation/'
        drift_decisions, acc_vector, drift_labels = np.loadtxt(
            path + str(file_name) + '.csv', delimiter=',', unpack=True, dtype=np.dtype(
                [('drift_decisions', 'bool'), ('acc_vector', 'int'), ('drift_labels', 'bool')]))

        evaluation_results["drift_decisions"] = drift_decisions
        evaluation_results["drift_labels"] = drift_labels
        evaluation_results["acc_vector"] = acc_vector

    elif change_detection_evaluation:
        path = '../Files_Results/' + str(result_folder) + '/ChangeDetectionEvaluation/'
        drift_decisions, drift_labels = np.loadtxt(
            path + str(file_name) + '.csv', delimiter=',', unpack=True,
            dtype=np.dtype([('drift_decisions', 'bool'), ('drift_labels', 'bool')]))

        evaluation_results["drift_decisions"] = drift_decisions
        evaluation_results["drift_labels"] = drift_labels

    elif proxy_evaluation:
        path = '../Files_Results/' + str(result_folder) + '/ProxyEvaluation/'
        drift_decisions, acc_vector = np.loadtxt(
            path + str(file_name) + '.csv', delimiter=',', unpack=True, dtype=np.dtype(
                [('drift_decisions', 'bool'), ('acc_vector', 'int')]))

        evaluation_results["acc_vector"] = acc_vector

    return evaluation_results

