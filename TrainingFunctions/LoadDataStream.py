import pickle
import sys
import numpy as np


def load_data_stream(dataset, path, separate_train_test_file, image_data, drift_labels_known, proxy_evaluation):
    """

    Parameters
    ----------
    dataset: Name of data set
    path: Path name
    separate_train_test_file: Boolean whether data are stored in separate train and test file
    image_data: Boolean whether data are image data
    drift_labels_known: Boolean whether drift labels are known, i.e. change detection evaluation
    proxy_evaluation: Boolean whether class labels exist, i.e. proxy evaluation

    Returns a data stream
    -------

    """
    if separate_train_test_file:
        if not proxy_evaluation:
            print("Error: Examined data with separate train and test file all have validation classifier!")
            sys.exit()
        if drift_labels_known:
            print("Error: Examined data with separate train and test file do all not have drift labels!")
            sys.exit()

        # Load data from train and test CSV file
        data_stream_train = np.loadtxt('../Datasets/' + str(path) + str(dataset) + '_TRAIN.data',
                                       delimiter=',')
        data_stream_test = np.loadtxt('../Datasets/' + str(path) + str(dataset) + '_TEST.data',
                                      delimiter=',')
        # Concatenate train and test data for equal pre-processing like for one data file
        data_stream = np.concatenate((data_stream_train, data_stream_test), axis=0)
    else:
        if image_data:
            if not drift_labels_known or not proxy_evaluation:
                print("Error: Examined image data all have validation classifier and drift labels are known!")
                sys.exit()

        if "pickle" in dataset:
            # Load data from PICKLE file
            with open('../Datasets/' + str(path) + str(dataset), 'rb') as handle:
                data_stream = pickle.load(handle)
        else:
            # Load data from CSV file
            data_stream = np.loadtxt('../Datasets/' + str(path) + str(dataset), delimiter=',')

    return data_stream
