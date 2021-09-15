import pandas as pd
import os


def load_with_iterations_results(file_name, result_folder):

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
    elif change_detection_evaluation:
        path = '../Files_Results/' + str(result_folder) + '/ChangeDetectionEvaluation/'
    elif proxy_evaluation:
        path = '../Files_Results/' + str(result_folder) + '/ProxyEvaluation/'

    evaluation_results = pd.read_csv(path + str(file_name) + '.csv', index_col=0)

    return evaluation_results

