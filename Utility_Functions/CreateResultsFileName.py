from datetime import datetime


def create_results_file_name(dataset, algorithm_name, drift_labels_known, proxy_evaluation, image_data):
    """

    Parameters
    ----------
    dataset: String with name of the dataset
    algorithm_name: String with name of the algorithm (used for folder name + description in file name)
    drift_labels_known: Boolean whether Boolean whether change detection evaluation is used
    proxy_evaluation: Boolean whether proxy evaluation was used
    image_data: Boolean whether image data is used

    Returns the name of the file
    -------

    """
    # Create file name
    date = datetime.today().strftime('%Y-%m-%d_%H.%M')
    if (drift_labels_known and proxy_evaluation) or image_data:
        folder = 'ChangeDetectionEvaluation_and_ProxyEvaluation/'
    elif drift_labels_known:
        folder = 'ChangeDetectionEvaluation/'
    elif proxy_evaluation:
        folder = 'ProxyEvaluation/'
    else:
        folder = 'No_Evaluation/'
    file_name = '../Files_Results/' + algorithm_name + '/' + str(folder) + algorithm_name + '_' + str(dataset) + '_' + date

    return file_name
