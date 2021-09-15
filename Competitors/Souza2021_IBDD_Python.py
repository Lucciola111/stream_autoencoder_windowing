import sys
import pandas as pd
import numpy as np
from Competitors.HelperFiles.Souza2021_IBDD_detectors import IBDD, plot_acc
from TrainingFunctions.LoadDataStream import load_data_stream
from TrainingFunctions.PreprocessingWithoutLabels import preprocessing_without_labels
from TrainingFunctions.PreprocessingWithLabels import preprocessing_with_labels
from EvaluationFunctions.GeneratePerformanceMetrics import generate_performance_metrics
from Evaluation.Plot_Accuracy import plot_accuracy
from Utility_Functions.CreateResultsFileName import create_results_file_name
from Utility_Functions.CreateResultsValues import create_results_values

# 1. Load Data
# Differentiate whether data is provided in separate train and test file
separate_train_test_file = False
image_data = True
drift_labels_known = True
proxy_evaluation = True

if not drift_labels_known and not proxy_evaluation:
    print("Error: Change detection evaluation and/or proxy evaluation missing!")
    sys.exit()

if not proxy_evaluation:
    acc_vector = False
if not drift_labels_known:
    drift_labels = False

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
        if proxy_evaluation:
            path = "Generated_Streams/Drift_And_Classifier_Labels/"

            dataset = "RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids_300MinL_2000MaxL_2021-08-06_10.57.pickle"
        else:
            path = "Generated_Streams/Drift_Labels/"

            # Experiments Evaluation
            dataset = "RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.01_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-10_10.32.pickle"
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
n_train_data = 500
n_val_data = n_train_data
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
    train_y = pd.Series(train_y)
    val_y = pd.Series(val_y)
    test_y = pd.Series(test_y)
else:
    train_X, val_X, test_X = preprocessing_without_labels(
        data_stream, n_instances, n_train_data, n_val_data, n_test_data)
    train_y, val_y, test_y = False, False, False
# Set number of dimensions
n_dimensions = train_X.shape[1]


# 3. Apply IBD drift detector
window_size = len(train_X)  # window parameter to build the images for comparison
epsilon = 3  # number of MSD values above/below threshold

train_X = pd.DataFrame(train_X)
test_X = pd.DataFrame(test_X)
[drift_decisions, acc_vector, mean_acc, time] = IBDD(train_X, test_X, window_size, epsilon, train_y, test_y)

# 4. Evaluation
# Fill array with indices of drift points
detected_drift_points = list(np.where(drift_decisions)[0])
# 4.1 Proxy Evaluation
if proxy_evaluation:
    # Calculate mean accuracy of classifier
    mean_acc = np.mean(acc_vector) * 100
    print('Average classification accuracy: {}%'.format(np.round(mean_acc, 2)))

# 4.2. Change detection evaluation
if drift_labels_known:
    # Define acceptance levels
    acceptance_levels = [60, 120, 180, 300]
    # Generate performance metrics
    simple_metrics = False
    complex_metrics = True
    performance_metrics = generate_performance_metrics(
        drift_decisions=drift_decisions, drift_labels=drift_labels, acceptance_levels=acceptance_levels,
        simple_metrics=simple_metrics, complex_metrics=complex_metrics, index_name='IBDD')

# 5. Save results in file
# Create file name
file_name = create_results_file_name(
    dataset=dataset, algorithm_name='Competitor_IBDD', drift_labels_known=drift_labels_known,
    proxy_evaluation=proxy_evaluation, image_data=image_data)
# Save according to selected evaluation values for drift_decisions, acc_vector, drift_labels
result_data, result_header, result_format = create_results_values(
    proxy_evaluation=proxy_evaluation, drift_labels_known=drift_labels_known, drift_decisions=drift_decisions,
    acc_vector=acc_vector, drift_labels=drift_labels)
# Save file
np.savetxt(str(file_name)
           + '_RESULTS'
           + '_windowSize' + str(window_size)
           + '_epsilon' + str(epsilon)
           + '.csv', result_data, header=result_header, delimiter=",", fmt=result_format)

# Plot accuracy and drift points
if proxy_evaluation:
    plot_file_name = "ProxyEvaluation_IBDD_" + str(dataset) + ".pdf"
    plot_accuracy(acc_vector=acc_vector, method_name='IBDD', drifts=detected_drift_points,
                  plot_file_name=plot_file_name, latex_font=True)
