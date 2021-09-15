import numpy as np
import os
import sys


def load_results_SAW_where(where_file_names, file_names, result_folder):

    proxy_and_change_detection_evaluation = os.path.exists(
        '../Files_Results/' + str(result_folder) + '/ChangeDetectionEvaluation_and_ProxyEvaluation/' + str(
            where_file_names[file_names[0]]) + '.csv')

    change_detection_evaluation = os.path.exists(
        '../Files_Results/' + str(result_folder) + '/ChangeDetectionEvaluation/' + str(
            where_file_names[file_names[0]]) + '.csv')

    if not proxy_and_change_detection_evaluation and not change_detection_evaluation:
        print("Results not available!")
        sys.exit()

    if proxy_and_change_detection_evaluation:
        path = '../Files_Results/SAW_Autoencoder_ADWIN_Training/ChangeDetectionEvaluation_and_ProxyEvaluation/'

    else:
        path = '../Files_Results/SAW_Autoencoder_ADWIN_Training/ChangeDetectionEvaluation/'

    all_errors_per_dimension = np.loadtxt(
        path + str(where_file_names[file_names[0]]) + '.csv', delimiter=',')

    all_errors_per_dimension_old_pattern = np.loadtxt(
        path + str(where_file_names[file_names[1]]) + '.csv', delimiter=',')

    return all_errors_per_dimension, all_errors_per_dimension_old_pattern

