import tensorflow as tf
import numpy as np
import sys
import pickle
from sklearn.ensemble import RandomForestClassifier
# from timeit import default_timer as timer
from time import process_time as timer
from TrainingFunctions.LoadDataStream import load_data_stream
from TrainingFunctions.PreprocessingWithLabels import preprocessing_with_labels
from TrainingFunctions.PreprocessingWithoutLabels import preprocessing_without_labels
from EvaluationFunctions.GenerateResultsTableIterations import generate_results_table_iterations
from Utility_Functions.CreateResultsFileName import create_results_file_name

# Only Proxy Evaluation possible
# Training of autoencoder not necessary


# 1. Load Data
# Differentiate whether data is provided in separate train and test file
separate_train_test_file = False
image_data = True
drift_labels_known = True
proxy_evaluation = True

if not proxy_evaluation:
    print("Error: Proxy evaluation missing!")
    sys.exit()

# 0. Set all parameters
n_iterations = 10

for dataset in datasets_image:

    if separate_train_test_file:
        path = "IBDD_Datasets/benchmark_real/"

        dataset = "Yoga"
        # dataset = "StarLightCurves"
        # dataset = "Heartbeats"
    elif image_data:
        path = "Generated_Streams/Image_Data_Drift_And_Classifier_Labels/"

        dataset = "RandomMNIST_and_FashionMNIST_SortAllNumbers19DR_2021-08-06_11.07.pickle"
    else:
        if drift_labels_known:
            path = "Generated_Streams/Drift_And_Classifier_Labels/"

            dataset = "RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids_300MinL_2000MaxL_2021-08-06_10.57.pickle"

        else:
            path = "Generated_Streams/Classifier_Labels/"
            dataset = ""

    # Load data stream
    data_stream = load_data_stream(dataset=dataset, path=path, separate_train_test_file=separate_train_test_file,
                                   image_data=image_data, drift_labels_known=drift_labels_known,
                                   proxy_evaluation=proxy_evaluation)

    # Set number of instances
    n_instances = data_stream.shape[0]
    # Set number of train data, validation data, and test data
    if dataset == "Yoga":
        n_train_data = 300
    elif dataset == "Heartbeats":
        n_train_data = 500
    else:
        n_train_data = 1000
    n_val_data = 0
    n_test_data = int(n_instances - n_val_data - n_train_data)
    tumbling_window_size = n_test_data

    # 2. Pre-processing
    # Separate data stream and drift labels
    if drift_labels_known:
        data_stream, drift_labels = data_stream[:, :-1], data_stream[:, -1]
        drift_labels = drift_labels[(len(drift_labels) - n_test_data):]
    # Preprocess data stream
    if proxy_evaluation:
        train_X, train_y, val_X, val_y, test_X, test_y = preprocessing_with_labels(
            data_stream, n_instances, n_train_data, n_val_data, n_test_data, image_data)
    else:
        train_X, val_X, test_X = preprocessing_without_labels(
            data_stream, n_instances, n_train_data, n_val_data, n_test_data)
    # Set number of dimensions
    n_dimensions = train_X.shape[1]

    # Initialize arrays for results
    all_accuracies = []
    all_performance_metrics = []
    all_times_per_example = []
    # Make several iterations
    for iteration in range(n_iterations):
        print("Global Iteration:")
        print(iteration)

        # 3. Train classifier for Evaluation
        if proxy_evaluation:
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
            model.fit(np.concatenate((train_X, val_X), axis=0), np.concatenate((train_y, val_y), axis=0))
            acc_vector = np.zeros(len(test_X), dtype=int)

        start = timer()
        for idx in range(0, n_test_data, tumbling_window_size):
            # Test: Make prediction for element with classifier
            if proxy_evaluation:
                new_element = test_X[idx: (idx + tumbling_window_size)]

                if proxy_evaluation:
                    y_pred = model.predict(new_element)
                    for element in range(len(new_element)):
                        if y_pred[element] == test_y[idx + element]:
                            acc_vector[idx + element] = 1

            if idx % 8000 == 0:
                print(f"Iteration of Loop: {idx}/{len(test_X)}")

        # 6. Evaluation
        # 6.1 Proxy Evaluation
        if proxy_evaluation:
            mean_acc = np.mean(acc_vector) * 100
            print('Average classification accuracy: {}%'.format(np.round(mean_acc, 2)))
            all_accuracies.append(mean_acc)
        # Measure the elapsed time
        end = timer()
        execution_time = end - start
        print('Time per example: {} sec'.format(np.round(execution_time / len(test_X), 4)))
        print('Total time: {} sec'.format(np.round(execution_time, 2)))
        all_times_per_example.append(execution_time / len(test_X))

    evaluation_results = generate_results_table_iterations(
        drift_labels_known=False, proxy_evaluation=proxy_evaluation,
        all_performance_metrics=all_performance_metrics, all_accuracies=all_accuracies,
        all_times_per_example=all_times_per_example)

    # 7. Save results in file from iterations
    # Create file name
    file_name = create_results_file_name(
        dataset=dataset, algorithm_name='Baseline_StaticClassifier', drift_labels_known=drift_labels_known,
        proxy_evaluation=proxy_evaluation, image_data=image_data)
    # Save results
    evaluation_results.to_csv(
        str(file_name)
        + '_' + str(n_iterations) + 'ITERATIONS_'
        + '.csv')
