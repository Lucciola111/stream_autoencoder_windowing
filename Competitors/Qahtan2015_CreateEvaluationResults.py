import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from TrainingFunctions.LoadDataStream import load_data_stream
from TrainingFunctions.PreprocessingWithoutLabels import preprocessing_without_labels
from TrainingFunctions.PreprocessingWithLabels import preprocessing_with_labels
from EvaluationFunctions.GeneratePerformanceMetrics import generate_performance_metrics
from Utility_Functions.CreateResultsFileName import create_results_file_name

# 0. Set parameter
window_size = 150
# Take care! Do also update selected dataset
experiment = "onlyMeanDrift_VerySmallVariance_50DR_RandomNormal_Min1DimsBroken"
# experiment = "onlyMeanDrift_SmallVariance_50DR_RandomNormal_Min1DimsBroken"
# experiment = "onlyMeanDrift_HigherVariance_50DR_RandomNormal_Min1DimsBroken"
# experiment = "onlyVarianceDrift_50DR_RandomNormal_Min1DimsBroken"
# experiment = "50DR_RandomNormal_Min100DimsBroken"
# experiment = "50DR_RandomNormal_Min1DimsBroken"
# experiment = "50DR_RandomRandomRBF_Min1DriftCentroid_0to1CS"

# 1. Load Data
# Differentiate whether data is provided in separate train and test file
separate_train_test_file = False
image_data = False
drift_labels_known = True
proxy_evaluation = False

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
            folder = 'ChangeDetectionEvaluation_and_ProxyEvaluation/'

            dataset = "RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids_300MinL_2000MaxL_2021-08-06_10.57.pickle"
        else:
            path = "Generated_Streams/Drift_Labels/"
            folder = 'ChangeDetectionEvaluation/'

            dataset = "RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.01_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-10_10.32.pickle"
            # dataset = "RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.05_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.42.pickle"
            # dataset = "RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.25_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.45.pickle"
            # dataset = "RandomNumpyRandomNormalUniform_onlyVarianceDrift_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_11.15.pickle"
            # dataset = "RandomNumpyRandomNormalUniform_50DR_100Dims_100MinDimBroken_300MinL_2000MaxL_2021-08-06_10.54.pickle"
            # dataset = "RandomNumpyRandomNormalUniform_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.53.pickle"

    else:
        if proxy_evaluation:
            path = "Generated_Streams/Classifier_Labels/"
            folder = 'ProxyEvaluation/'

            dataset = "RandomRBFGenerator_DR1_i4000_ni2000_cs2.87_dc30.csv"
        else:
            path = "Generated_Streams/No_Labels/"

# Load data stream
data_stream = load_data_stream(dataset=dataset, path=path, separate_train_test_file=separate_train_test_file,
                               image_data=image_data, drift_labels_known=drift_labels_known,
                               proxy_evaluation=proxy_evaluation)
# Set number of instances
n_instances = data_stream.shape[0]
# Set number of train data, validation data, and test data
n_train_data = 1000
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
else:
    train_X, val_X, test_X = preprocessing_without_labels(
        data_stream, n_instances, n_train_data, n_val_data, n_test_data)

# Set number of dimensions
n_dimensions = train_X.shape[1]

# 3. Train classifier for Evaluation
if proxy_evaluation:
    model_classifier = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
    model_classifier.fit(np.concatenate((train_X, val_X), axis=0), np.concatenate((train_y, val_y), axis=0))
    acc_vector = np.zeros(len(test_y), dtype=int)

# 4. Load drift decisions
drift_decision_files = {'onlyMeanDrift_VerySmallVariance_50DR_RandomNormal_Min1DimsBroken': 'output_onlyMeanDrift_VerySmallVariance_50DR_RandomNormal_Min1DimsBroken_windowSize150',
                        'onlyMeanDrift_SmallVariance_50DR_RandomNormal_Min1DimsBroken': 'output_onlyMeanDrift_SmallVariance_50DR_RandomNormal_Min1DimsBroken_windowSize150',
                        'onlyMeanDrift_HigherVariance_50DR_RandomNormal_Min1DimsBroken': 'output_onlyMeanDrift_HigherVariance_50DR_RandomNormal_Min1DimsBroken_windowSize150',
                        'onlyVarianceDrift_50DR_RandomNormal_Min1DimsBroken': 'output_onlyVarianceDrift_50DR_RandomNormal_Min1DimsBroken_windowSize150',
                        '50DR_RandomNormal_Min100DimsBroken': 'output_50DR_RandomNormal_Min100DimsBroken_windowSize150',
                        '50DR_RandomNormal_Min1DimsBroken': 'output_50DR_RandomNormal_Min1DimsBroken_windowSize150',
                        '50DR_RandomRandomRBF_Min1DriftCentroid_0to1CS': 'output_50DR_RandomRandomRBF_Min1DriftCentroid_0to1CS_windowSize150'
                        }

drift_decisions_PCA_CD = np.loadtxt(
    '../Files_Results/Competitor_PCA-CD/' + str(folder) + drift_decision_files[
        experiment] + '.txt')
# Select only drifts after the start of the test data
drift_decisions_PCA_CD = drift_decisions_PCA_CD[drift_decisions_PCA_CD > (n_train_data + n_val_data)]

# 5. Retrain model after drift was detected
if proxy_evaluation:
    for idx in range(len(test_X)):
        # Test: Make prediction for element with classifier
        y_pred = model_classifier.predict(test_X[idx].reshape(1, -1))
        if y_pred == test_y[idx]:
            acc_vector[idx] = 1
        # If drift is detected
        if idx in drift_decisions_PCA_CD:
            print(f"Change detected at index {idx}")
            if proxy_evaluation:
                # Train model again on new window data
                window_train_X = test_X[(idx - window_size):idx]
                window_train_y = test_y[(idx - window_size):idx]
                model_classifier.fit(window_train_X, window_train_y)

# 5.1. Change detection evaluation
if drift_labels_known:
    # Define acceptance levels
    acceptance_levels = [60, 120, 180, 300]
    # Generate performance metrics
    simple_metrics = False
    complex_metrics = True
    performance_metrics = generate_performance_metrics(
        drift_decisions=drift_decisions_PCA_CD, drift_labels=drift_labels, acceptance_levels=acceptance_levels,
        simple_metrics=simple_metrics, complex_metrics=complex_metrics, index_name='PCA-CD')
    evaluation_results = performance_metrics

# 5.1. Proxy evaluation
if proxy_evaluation:
    # Calculate mean accuracy of classifier
    mean_acc = np.mean(acc_vector) * 100
    print('Average classification accuracy: {}%'.format(np.round(mean_acc, 2)))
    if drift_labels_known:
        evaluation_results["Accuracy"] = mean_acc
    else:
        evaluation_results = pd.DataFrame(data=mean_acc, index=["PCA-CD"])

# Add dummy time
evaluation_results["Time per Example"] = 0

# 7. Save results in file from iterations
# Create file name
file_name = create_results_file_name(
    dataset=dataset, algorithm_name='Competitor_PCA-CD', drift_labels_known=drift_labels_known,
    proxy_evaluation=proxy_evaluation, image_data=image_data)
# Save file
evaluation_results.to_csv(
    str(file_name)
    + '_withoutITERATIONS_'
    + str(window_size) + 'windowSize'
    + '.csv')
