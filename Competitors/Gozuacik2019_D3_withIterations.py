import sys
import numpy as np
from river.drift import D3
from time import process_time as timer
from sklearn.ensemble import RandomForestClassifier
from TrainingFunctions.LoadDataStream import load_data_stream
from TrainingFunctions.PreprocessingWithoutLabels import preprocessing_without_labels
from TrainingFunctions.PreprocessingWithLabels import preprocessing_with_labels
from EvaluationFunctions.GeneratePerformanceMetrics import generate_performance_metrics
from EvaluationFunctions.GenerateResultsTableIterations import generate_results_table_iterations
from Utility_Functions.CreateResultsFileName import create_results_file_name

# Not possible to execute on server!
# Installation using GitHub necessary: https://github.com/ogozuacik/d3-discriminative-drift-detector-concept-drift

# Set parameters
n_iterations = 10

# 1. Load Data
# Differentiate whether data is provided in separate train and test file
separate_train_test_file = False
image_data = False
drift_labels_known = True
proxy_evaluation = False

if not drift_labels_known and not proxy_evaluation:
    print("Error: Change detection evaluation and/or proxy evaluation missing!")
    sys.exit()

if not proxy_evaluation:
    acc_vector = False
if not drift_labels_known:
    drift_labels = False

# Set name of data set and path
if separate_train_test_file:
    path = "IBDD_Datasets/benchmark_real/"

    # dataset = "Yoga"
    # dataset = "StarLightCurves"
    # dataset = "Heartbeats"
elif image_data:
    path = "Generated_Streams/Image_Data_Drift_And_Classifier_Labels/"

    # dataset = "RandomMNIST_and_FashionMNIST_SortAllNumbers19DR_2021-08-06_11.07.pickle"
else:
    if drift_labels_known:
        if proxy_evaluation:
            path = "Generated_Streams/Drift_And_Classifier_Labels/"

            # dataset = "RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids_300MinL_2000MaxL_2021-08-06_10.57.pickle"
        else:
            path = "Generated_Streams/Drift_Labels/"

            # Experiments Evaluation
            # dataset = "RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.01_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-10_10.32.pickle"
            # dataset = "RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.05_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.42.pickle"
            # dataset = "RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.25_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.45.pickle"
            # dataset = "RandomNumpyRandomNormalUniform_onlyVarianceDrift_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_11.15.pickle"
            # dataset = "RandomNumpyRandomNormalUniform_50DR_100Dims_100MinDimBroken_300MinL_2000MaxL_2021-08-06_10.54.pickle"
            # dataset = "RandomNumpyRandomNormalUniform_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.53.pickle"
            # dataset = "Mixed_300MinDistance_DATASET_A_RandomNumpyRandomNormalUniform_50DR_100Dims_1MinDimBroken_DATASET_B_RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids.pickle"

            # Experiments Time Complexity
            # "Time_RandomNumpyRandomNormalUniform_10DR_10Dims_1MinDimBroken_300MinL_2000MaxL_2021-09-06_22.24.pickle",
            # "Time_RandomNumpyRandomNormalUniform_10DR_50Dims_5MinDimBroken_300MinL_2000MaxL_2021-09-06_22.25.pickle",
            # "Time_RandomNumpyRandomNormalUniform_10DR_100Dims_10MinDimBroken_300MinL_2000MaxL_2021-09-06_22.26.pickle",
            # "Time_RandomNumpyRandomNormalUniform_10DR_500Dims_50MinDimBroken_300MinL_2000MaxL_2021-09-06_22.26.pickle",
            # "Time_RandomNumpyRandomNormalUniform_10DR_1000Dims_100MinDimBroken_300MinL_2000MaxL_2021-09-06_22.27.pickle"

    else:
        path = "Generated_Streams/Classifier_Labels/"
        dataset = ""

print("Current dataset:")
print(dataset)

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

# Start global iterations
all_performance_metrics = []
all_accuracies = []
all_times_per_example = []
for iteration in range(n_iterations):
    print("Global Iteration:")
    print(iteration)

    # 3. Train classifier for Evaluation
    if proxy_evaluation:
        model_classifier = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
        model_classifier.fit(np.concatenate((train_X, val_X), axis=0), np.concatenate((train_y, val_y), axis=0))
        acc_vector = np.zeros(len(test_y), dtype=int)

    # 4. Drift Detection with D3
    start = timer()

    d3 = D3()
    # Transform data to array with dictionaries
    data_stream_dict = []
    for idx in range(len(train_X)):
        new_element = dict(enumerate(train_X[idx], 1))
        data_stream_dict.append(new_element)
    for idx in range(len(val_X)):
        new_element = dict(enumerate(val_X[idx], 1))
        data_stream_dict.append(new_element)
    for idx in range(len(test_X)):
        new_element = dict(enumerate(test_X[idx], 1))
        data_stream_dict.append(new_element)

    # Start saving drift decisions when test data start
    drift_decisions = [False] * len(data_stream_dict)

    for idx in range(len(data_stream_dict)):
        test_idx = idx - (n_train_data + n_val_data)
        # If test data start and proxy evaluation, start prediction
        if test_idx >= 0 and proxy_evaluation:
            # Test: Make prediction for element with classifier
            y_pred = model_classifier.predict(test_X[test_idx].reshape(1, -1))
            if y_pred == test_y[test_idx]:
                acc_vector[test_idx] = 1
        # Detect drift
        in_drift, in_warning = d3.update(data_stream_dict[idx])
        # If drift is detected
        if in_drift:
            print(f"Change detected at index {idx}")
            drift_decisions[idx] = True
            if test_idx >= d3.new_data_window_size and proxy_evaluation:
                # Train model again on new window data
                window_train_X = test_X[(test_idx - d3.new_data_window_size):test_idx]
                window_train_y = test_y[(test_idx - d3.new_data_window_size):test_idx]
                model_classifier.fit(window_train_X, window_train_y)

    # Save only drift decisions for test data
    drift_decisions_in_test_data = drift_decisions[(n_train_data + n_val_data):]

    # Measure the elapsed time
    end = timer()
    execution_time = end - start
    print('Time per example: {} sec'.format(np.round(execution_time / len(test_X), 4)))
    print('Total time: {} sec'.format(np.round(execution_time, 2)))
    all_times_per_example.append(execution_time / len(test_X))

    # 5. Evaluation

    # 5.1 Proxy Evaluation
    if proxy_evaluation:
        # Calculate mean accuracy of classifier
        mean_acc = np.mean(acc_vector) * 100
        print('Average classification accuracy: {}%'.format(np.round(mean_acc, 2)))
        all_accuracies.append(mean_acc)

    # 5.2. Change detection evaluation
    if drift_labels_known:
        # Define acceptance levels
        acceptance_levels = [60, 120, 180, 300]
        # Generate performance metrics
        simple_metrics = False
        complex_metrics = True
        performance_metrics = generate_performance_metrics(
            drift_decisions=drift_decisions_in_test_data, drift_labels=drift_labels,
            acceptance_levels=acceptance_levels,
            simple_metrics=simple_metrics, complex_metrics=complex_metrics, index_name='D3')
        all_performance_metrics.append(performance_metrics)

# Generate evaluation results table
evaluation_results = generate_results_table_iterations(
    drift_labels_known=drift_labels_known, proxy_evaluation=proxy_evaluation,
    all_performance_metrics=all_performance_metrics, all_accuracies=all_accuracies,
    all_times_per_example=all_times_per_example)

# 7. Save results in file from iterations
# Create file name
file_name = create_results_file_name(
    dataset=dataset, algorithm_name='Competitor_D3', drift_labels_known=drift_labels_known,
    proxy_evaluation=proxy_evaluation, image_data=image_data)
# Save file
evaluation_results.to_csv(
    str(file_name)
    + '_' + str(n_iterations) + 'ITERATIONS_'
    + '.csv')
