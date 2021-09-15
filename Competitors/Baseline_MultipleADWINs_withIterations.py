import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from time import process_time as timer
from skmultiflow.drift_detection.adwin import ADWIN
from TrainingFunctions.LoadDataStream import load_data_stream
from TrainingFunctions.PreprocessingWithLabels import preprocessing_with_labels
from TrainingFunctions.PreprocessingWithoutLabels import preprocessing_without_labels
from TrainingFunctions.DetectDriftMultipleADWINSOnline import detect_drift_multiple_adwins_online
from EvaluationFunctions.GeneratePerformanceMetrics import generate_performance_metrics
from EvaluationFunctions.GenerateResultsTableIterations import generate_results_table_iterations
from Utility_Functions.CreateResultsFileName import create_results_file_name
from Utility_Functions.GPU import setup_machine

# Use GPU
setup_machine(cuda_device=2)

# 0. Set all parameters
n_iterations = 10
# Set agreement rate for drift detection
agreement_rate = 0.1
# Set option for initializing ADWIN after a detected drift
reinitialize_adwin = False

# 1. Load Data
# Differentiate whether data is provided in separate train and test file
separate_train_test_file = False
image_data = False
drift_labels_known = True
proxy_evaluation = False

if not drift_labels_known and not proxy_evaluation:
    print("Error: Change detection evaluation and/or proxy evaluation missing!")
    sys.exit()

# Initialize parameters
if not proxy_evaluation:
    acc_vector = False
if not drift_labels_known:
    drift_labels = False

# Load path and name of dataset
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

    start = timer()

    # 4. Drift Detection:
    # Test: Apply n ADWINs on n dimensions
    # Train: Retrain classifier if >=agreement_threshold dimensions detect a drift
    agreement_threshold = int(agreement_rate * n_dimensions)

    adwins = {}
    widths_multiple_adwins = []
    drift_decisions_fusion = []
    drift_points_multiple_adwins = []

    for dim in range(n_dimensions):
        # Initialize ADWINs
        adwins[dim] = ADWIN(delta=0.002)

    for idx in range(n_test_data):
        # If we have validation classifier: Set parameters for retraining classifier after drift
        model_classifier_parameter = model_classifier if proxy_evaluation else False
        test_y_parameter = test_y if proxy_evaluation else False

        # Test: Make prediction for element with classifier
        if proxy_evaluation:
            y_pred = model_classifier.predict(test_X[idx].reshape(1, -1))
            if y_pred == test_y[idx]:
                acc_vector[idx] = 1

        # Detect Drift with n ADWINs for n dimensions
        adwins, adwins_width, drift_decision_fusion, drift_point_dims = detect_drift_multiple_adwins_online(
            test_X=test_X, idx=idx, adwins=adwins, agreement_threshold=agreement_threshold,
            reinitialize_adwin=reinitialize_adwin, model=model_classifier_parameter, test_y=test_y_parameter)

        widths_multiple_adwins = adwins_width if len(widths_multiple_adwins) == 0 else np.column_stack(
            (widths_multiple_adwins, adwins_width))
        drift_decisions_fusion.append(drift_decision_fusion)
        drift_points_multiple_adwins.append(drift_point_dims)

        if idx % 8000 == 0:
            print(f"Iteration of Loop: {idx}/{len(test_X)}")

    # 5. Evaluation
    # 5.1 Drift Detection Evaluation
    if drift_labels_known:
        # Define acceptance levels
        acceptance_levels = [60, 120, 180, 300]
        # Generate performance metrics
        simple_metrics = False
        complex_metrics = True
        performance_metrics = generate_performance_metrics(
            drift_decisions=drift_decisions_fusion, drift_labels=drift_labels, acceptance_levels=acceptance_levels,
            simple_metrics=simple_metrics, complex_metrics=complex_metrics, index_name='ADWIN-' + str(int(agreement_rate*100)))
        all_performance_metrics.append(performance_metrics)

    # 5.2 Proxy Evaluation
    if proxy_evaluation:
        # Calculate mean accuracy of classifier
        mean_acc = np.mean(acc_vector) * 100
        print('Average classification accuracy: {}%'.format(np.round(mean_acc, 2)))
        all_accuracies.append(mean_acc)
    # Measure the elapsed time
    end = timer()
    execution_time = end - start
    print('Time per example: {} sec'.format(np.round(execution_time / len(test_X), 4)))
    print('Total time: {} sec'.format(np.round(execution_time, 2)))
    all_times_per_example.append(execution_time / len(test_X))

# Generate evaluation results table
evaluation_results = generate_results_table_iterations(
    drift_labels_known=drift_labels_known, proxy_evaluation=proxy_evaluation,
    all_performance_metrics=all_performance_metrics, all_accuracies=all_accuracies,
    all_times_per_example=all_times_per_example)

# 7. Save results in file from iterations
# Create file name
file_name = create_results_file_name(
    dataset=dataset, algorithm_name='Baseline_MultipleADWINs', drift_labels_known=drift_labels_known,
    proxy_evaluation=proxy_evaluation, image_data=image_data)
# Save file
evaluation_results.to_csv(
    str(file_name)
    + '_' + str(n_iterations) + 'ITERATIONS_'
    + '_agreementRate' + str(agreement_rate)
    + '_reinitializeADWIN' + str(reinitialize_adwin)
    + '.csv')
