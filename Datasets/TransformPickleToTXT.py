import numpy as np
import pickle

drift_labels = False
drift_labels_and_classifier_labels = False
image_data_drift_labels_and_classifier_labels = False
separate_train_test_file = True

if drift_labels:
    path = './Generated_Streams/Drift_Labels/'
    dataset = "RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.01_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-10_10.32.pickle"
    # dataset = "RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.05_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.42.pickle"
    # dataset = "RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.25_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.45.pickle"
    # dataset = "RandomNumpyRandomNormalUniform_onlyVarianceDrift_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_11.15.pickle"
    # dataset = "RandomNumpyRandomNormalUniform_50DR_100Dims_100MinDimBroken_300MinL_2000MaxL_2021-08-06_10.54.pickle"
    # dataset = "RandomNumpyRandomNormalUniform_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.53.pickle"
    # dataset = "Mixed_300MinDistance_DATASET_A_RandomNumpyRandomNormalUniform_50DR_100Dims_1MinDimBroken_DATASET_B_RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids.pickle"

if drift_labels_and_classifier_labels:
    path = "Generated_Streams/Drift_And_Classifier_Labels/"
    dataset = "RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids_300MinL_2000MaxL_2021-08-06_10.57.pickle"

if image_data_drift_labels_and_classifier_labels:
    path = "Generated_Streams/Image_Data_Drift_And_Classifier_Labels/"
    dataset = "RandomMNIST_and_FashionMNIST_SortAllNumbers19DR_2021-08-06_11.07.pickle"

if drift_labels_and_classifier_labels or image_data_drift_labels_and_classifier_labels:
    with open(str(path) + str(dataset), 'rb') as handle:
        data_stream = pickle.load(handle)

if separate_train_test_file:
    # Load data from train and test CSV file
    path = "IBDD_Datasets/benchmark_real/"
    dataset = "Yoga"
    # dataset = "StarLightCurves"
    # dataset = "Heartbeats"

    data_stream_train = np.loadtxt(str(path) + str(dataset) + '_TRAIN.data',
                                   delimiter=',')
    data_stream_test = np.loadtxt(str(path) + str(dataset) + '_TEST.data',
                                  delimiter=',')
    # Concatenate train and test data for equal pre-processing like for one data file
    data_stream = np.concatenate((data_stream_train, data_stream_test), axis=0)


if drift_labels or separate_train_test_file:
    data_stream_X = data_stream[:, :-1]
else:
    data_stream_X = data_stream[:, :-2]

# Save data stream to txt
# np.savetxt(str(path) + 'Mixed_300MinDistance_DATASET_A_RandomNumpyRandomNormalUniform_DATASET_B_RandomRandomRBF' + '_withoutLabel.txt',
np.savetxt(str(path) + str(dataset) + '_withoutLabel.txt',
           data_stream_X, delimiter=" ")

# Qahtan:
# ./CD ../../../../../Code_ChangeDetection_in_HDD/Datasets/Generated_Streams/Drift_Labels/RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.01_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-10_10.32.pickle_withoutLabel.txt 150 500 output_onlyMeanDrift_VerySmallVariance_50DR_RandomNormal_Min1DimsBroken_windowSize150.txt 100 0.005 1
# ./CD ../../../../../Code_ChangeDetection_in_HDD/Datasets/Generated_Streams/Drift_Labels/RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.05_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.42.pickle_withoutLabel.txt 150 500 output_onlyMeanDrift_SmallVariance_50DR_RandomNormal_Min1DimsBroken_windowSize150.txt 100 0.005 1
# ./CD ../../../../../Code_ChangeDetection_in_HDD/Datasets/Generated_Streams/Drift_Labels/RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.25_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.45.pickle_withoutLabel.txt 150 500 output_onlyMeanDrift_HigherVariance_50DR_RandomNormal_Min1DimsBroken_windowSize150.txt 100 0.005 1
# ./CD ../../../../../Code_ChangeDetection_in_HDD/Datasets/Generated_Streams/Drift_Labels/RandomNumpyRandomNormalUniform_onlyVarianceDrift_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_11.15.pickle_withoutLabel.txt 150 500 output_onlyVarianceDrift_50DR_RandomNormal_Min1DimsBroken_windowSize150.txt 100 0.005 1
# ./CD ../../../../../Code_ChangeDetection_in_HDD/Datasets/Generated_Streams/Drift_Labels/RandomNumpyRandomNormalUniform_50DR_100Dims_100MinDimBroken_300MinL_2000MaxL_2021-08-06_10.54.pickle_withoutLabel.txt 150 500 output_50DR_RandomNormal_Min100DimsBroken_windowSize150.txt 100 0.005 1
# ./CD ../../../../../Code_ChangeDetection_in_HDD/Datasets/Generated_Streams/Drift_Labels/RandomNumpyRandomNormalUniform_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.53.pickle_withoutLabel.txt 150 500 output_50DR_RandomNormal_Min1DimsBroken_windowSize150.txt 100 0.005 1

# Qahtan not working for higher dimensions:
# X <= 100: working;
# 100 < X: Segmentation fault;
# 209 < X: error: terminate called after throwing an instance of 'std::domain_error'
#           what():  Record has the wrong size: 0
#           Aborted

# Not possible
# ./CD ../../../../../Code_ChangeDetection_in_HDD/Datasets/Generated_Streams/Drift_Labels/Mixed_300MinDistance_DATASET_A_RandomNumpyRandomNormalUniform_DATASET_B_RandomRandomRBF_withoutLabel.txt 150 500 output_100DR_Merged_Min1Broken_windowSize150.txt 200 0.005 1
# ./CD ../../../../../Code_ChangeDetection_in_HDD/Datasets/Generated_Streams/Drift_And_Classifier_Labels/RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids_300MinL_2000MaxL_2021-08-06_10.57.pickle_withoutLabel.txt 150 500 output_50DR_RandomRandomRBF_Min1DriftCentroid_0to1CS_windowSize150.txt 100 0.005 1
# ./CD ../../../../../Code_ChangeDetection_in_HDD/Datasets/Generated_Streams/Image_Data_Drift_And_Classifier_Labels/RandomMNIST_and_FashionMNIST_SortAllNumbers19DR_2021-08-06_11.07.pickle_withoutLabel.txt 150 500 output_50DR_RandomMNIST_FashionMNIST_SortAllNumbers_windowSize150.txt 784 0.005 1
# ./CD ../../../../../Code_ChangeDetection_in_HDD/Datasets/IBDD_Datasets/benchmark_real/Yoga_withoutLabel.txt 150 500 output_Yoga_windowSize150.txt 426 0.005 1
# ./CD ../../../../../Code_ChangeDetection_in_HDD/Datasets/IBDD_Datasets/benchmark_real/StarLightCurves_withoutLabel.txt 150 500 output_StarLightCurves_windowSize150.txt 1024 0.005 1
# ./CD ../../../../../Code_ChangeDetection_in_HDD/Datasets/IBDD_Datasets/benchmark_real/Heartbeats_withoutLabel.txt 150 500 output_Heartbeats_windowSize150.txt 280 0.005 1
