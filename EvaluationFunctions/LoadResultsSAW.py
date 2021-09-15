import numpy as np
import os


def load_results_SAW(file_name, result_folder):

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
        path = '../Files_Results/SAW_Autoencoder_ADWIN_Training/ChangeDetectionEvaluation_and_ProxyEvaluation/'
        widths, reconstruction_error_time, drift_decisions, acc_vector, drift_labels = np.loadtxt(
            path + str(file_name) + '.csv', delimiter=',', unpack=True, dtype=np.dtype(
                [('widths', 'int'), ('reconstruction_error_time', 'float'), ('drift_decisions', 'bool'),
                 ('acc_vector', 'int'), ('drift_labels', 'bool')]))

        evaluation_results["drift_decisions"] = drift_decisions
        evaluation_results["drift_labels"] = drift_labels
        evaluation_results["acc_vector"] = acc_vector


    elif change_detection_evaluation:
        path = '../Files_Results/SAW_Autoencoder_ADWIN_Training/ChangeDetectionEvaluation/'
        widths, reconstruction_error_time, drift_decisions, drift_labels = np.loadtxt(
            path + str(file_name) + '.csv', delimiter=',', unpack=True,
            dtype=np.dtype([('widths', 'int'), ('reconstruction_error_time', 'float'), ('drift_decisions', 'bool'),
                            ('drift_labels', 'bool')]))

        evaluation_results["drift_decisions"] = drift_decisions
        evaluation_results["drift_labels"] = drift_labels

    elif proxy_evaluation:
        path = '../Files_Results/SAW_Autoencoder_ADWIN_Training/ProxyEvaluation/'
        widths, reconstruction_error_time, drift_decisions, acc_vector = np.loadtxt(
            path + str(file_name) + '.csv', delimiter=',', unpack=True,
            dtype=np.dtype([('widths', 'int'), ('reconstruction_error_time', 'float'), ('drift_decisions', 'bool')]))

        evaluation_results["acc_vector"] = acc_vector

    return evaluation_results

