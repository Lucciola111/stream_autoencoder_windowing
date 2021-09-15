import sys


def load_framework_designs_filenames(experiment):
    """

    Parameters
    ----------
    experiment: Identifier of experiment for evaluation

    Returns the path and name of the dataset and a list with the file names of the results
    -------

    """
    result_file_names = {}

    # Experiments normal distribution
    if experiment == 'onlyMeanDrift_VerySmallVariance_50DR_RandomNormal_Min1DimsBroken':
        path = "Generated_Streams/Drift_Labels/"
        dataset = "RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.01_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-10_10.32.pickle"

        result_file_names["FILE_TrainNewAE_KeepADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.01_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-10_10.32.pickle_2021-08-12_00.57_10ITERATIONS__fitNewAETrue_fitFalse_initNewADWINFalse_feedNewADWINFalse'
        result_file_names["FILE_TrainNewAE_InitializeADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.01_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-10_10.32.pickle_2021-08-12_01.20_10ITERATIONS__fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_TrainNewAE_InitializeAndFeedADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.01_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-10_10.32.pickle_2021-08-12_10.40_10ITERATIONS__fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINTrue'
        result_file_names["FILE_RetrainAE_KeepADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.01_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-10_10.32.pickle_2021-08-12_11.07_10ITERATIONS__fitNewAEFalse_fitTrue_initNewADWINFalse_feedNewADWINFalse'
        result_file_names["FILE_RetrainAE_InitializeADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.01_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-10_10.32.pickle_2021-08-12_11.27_10ITERATIONS__fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_RetrainAE_InitializeAndFeedADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.01_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-10_10.32.pickle_2021-08-12_11.48_10ITERATIONS__fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINTrue'

    elif experiment == 'onlyMeanDrift_SmallVariance_50DR_RandomNormal_Min1DimsBroken':
        path = "Generated_Streams/Drift_Labels/"
        dataset = "SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.05_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.42.pickle"

        result_file_names["FILE_TrainNewAE_KeepADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.05_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.42.pickle_2021-08-13_01.57_10ITERATIONS__fitNewAETrue_fitFalse_initNewADWINFalse_feedNewADWINFalse'
        result_file_names["FILE_TrainNewAE_InitializeADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.05_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.42.pickle_2021-08-13_02.22_10ITERATIONS__fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_TrainNewAE_InitializeAndFeedADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.05_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.42.pickle_2021-08-13_02.59_10ITERATIONS__fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINTrue'
        result_file_names["FILE_RetrainAE_KeepADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.05_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.42.pickle_2021-08-13_03.48_10ITERATIONS__fitNewAEFalse_fitTrue_initNewADWINFalse_feedNewADWINFalse'
        result_file_names["FILE_RetrainAE_InitializeADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.05_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.42.pickle_2021-08-13_04.17_10ITERATIONS__fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_RetrainAE_InitializeAndFeedADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.05_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.42.pickle_2021-08-13_04.54_10ITERATIONS__fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINTrue'

    elif experiment == 'onlyMeanDrift_HigherVariance_50DR_RandomNormal_Min1DimsBroken':
        path = "Generated_Streams/Drift_Labels/"
        dataset = "SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.25_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.45.pickle"

        result_file_names["FILE_TrainNewAE_KeepADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.25_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.45.pickle_2021-08-13_06.13_10ITERATIONS__fitNewAETrue_fitFalse_initNewADWINFalse_feedNewADWINFalse'
        result_file_names["FILE_TrainNewAE_InitializeADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.25_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.45.pickle_2021-08-13_06.46_10ITERATIONS__fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_TrainNewAE_InitializeAndFeedADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.25_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.45.pickle_2021-08-13_08.41_10ITERATIONS__fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINTrue'
        result_file_names["FILE_RetrainAE_KeepADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.25_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.45.pickle_2021-08-13_10.57_10ITERATIONS__fitNewAEFalse_fitTrue_initNewADWINFalse_feedNewADWINFalse'
        result_file_names["FILE_RetrainAE_InitializeADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.25_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.45.pickle_2021-08-13_11.48_10ITERATIONS__fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_RetrainAE_InitializeAndFeedADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.25_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.45.pickle_2021-08-13_15.53_10ITERATIONS__fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINTrue'

    elif experiment == 'onlyVarianceDrift_50DR_RandomNormal_Min1DimsBroken':
        path = "Generated_Streams/Drift_Labels/"
        dataset = "RandomNumpyRandomNormalUniform_onlyVarianceDrift_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_11.15.pickle"

        result_file_names["FILE_TrainNewAE_KeepADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyVarianceDrift_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_11.15.pickle_2021-08-08_19.34_10ITERATIONS__fitNewAETrue_fitFalse_initNewADWINFalse_feedNewADWINFalse'
        result_file_names["FILE_TrainNewAE_InitializeADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyVarianceDrift_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_11.15.pickle_2021-08-08_20.07_10ITERATIONS__fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_TrainNewAE_InitializeAndFeedADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyVarianceDrift_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_11.15.pickle_2021-08-08_21.38_10ITERATIONS__fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINTrue'
        result_file_names["FILE_RetrainAE_KeepADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyVarianceDrift_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_11.15.pickle_2021-08-08_23.52_10ITERATIONS__fitNewAEFalse_fitTrue_initNewADWINFalse_feedNewADWINFalse'
        result_file_names["FILE_RetrainAE_InitializeADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyVarianceDrift_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_11.15.pickle_2021-08-09_00.40_10ITERATIONS__fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_RetrainAE_InitializeAndFeedADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyVarianceDrift_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_11.15.pickle_2021-08-09_02.41_10ITERATIONS__fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINTrue'

    elif experiment == '50DR_RandomNormal_Min100DimsBroken':
        path = "Generated_Streams/Drift_Labels/"
        dataset = "RandomNumpyRandomNormalUniform_50DR_100Dims_100MinDimBroken_300MinL_2000MaxL_2021-08-06_10.54.pickle"

        result_file_names["FILE_TrainNewAE_KeepADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_50DR_100Dims_100MinDimBroken_300MinL_2000MaxL_2021-08-06_10.54.pickle_2021-08-09_04.49_10ITERATIONS__fitNewAETrue_fitFalse_initNewADWINFalse_feedNewADWINFalse'
        result_file_names["FILE_TrainNewAE_InitializeADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_50DR_100Dims_100MinDimBroken_300MinL_2000MaxL_2021-08-06_10.54.pickle_2021-08-09_05.24_10ITERATIONS__fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_TrainNewAE_InitializeAndFeedADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_50DR_100Dims_100MinDimBroken_300MinL_2000MaxL_2021-08-06_10.54.pickle_2021-08-09_06.26_10ITERATIONS__fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINTrue'
        result_file_names["FILE_RetrainAE_KeepADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_50DR_100Dims_100MinDimBroken_300MinL_2000MaxL_2021-08-06_10.54.pickle_2021-08-09_08.23_10ITERATIONS__fitNewAEFalse_fitTrue_initNewADWINFalse_feedNewADWINFalse'
        result_file_names["FILE_RetrainAE_InitializeADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_50DR_100Dims_100MinDimBroken_300MinL_2000MaxL_2021-08-06_10.54.pickle_2021-08-09_09.01_10ITERATIONS__fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_RetrainAE_InitializeAndFeedADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_50DR_100Dims_100MinDimBroken_300MinL_2000MaxL_2021-08-06_10.54.pickle_2021-08-09_10.28_10ITERATIONS__fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINTrue'

    elif experiment == '50DR_RandomNormal_Min1DimsBroken':
        path = "Generated_Streams/Drift_Labels/"
        dataset = "RandomNumpyRandomNormalUniform_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.53.pickle""RandomNumpyRandomNormalUniform_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.53.pickle"

        result_file_names["FILE_TrainNewAE_KeepADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.53.pickle_2021-08-09_12.16_10ITERATIONS__fitNewAETrue_fitFalse_initNewADWINFalse_feedNewADWINFalse'
        result_file_names["FILE_TrainNewAE_InitializeADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.53.pickle_2021-08-09_12.50_10ITERATIONS__fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_TrainNewAE_InitializeAndFeedADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.53.pickle_2021-08-09_14.03_10ITERATIONS__fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINTrue'
        result_file_names["FILE_RetrainAE_KeepADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.53.pickle_2021-08-09_16.18_10ITERATIONS__fitNewAEFalse_fitTrue_initNewADWINFalse_feedNewADWINFalse'
        result_file_names["FILE_RetrainAE_InitializeADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.53.pickle_2021-08-09_17.06_10ITERATIONS__fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_RetrainAE_InitializeAndFeedADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.53.pickle_2021-08-12_11.48_10ITERATIONS__fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINTrue'

    # Experiments other datasets
    elif experiment == '100DR_Merged_Min1Broken':
        path = "Generated_Streams/Drift_Labels/"
        dataset = "Mixed_300MinDistance_DATASET_A_RandomNumpyRandomNormalUniform_50DR_100Dims_1MinDimBroken_DATASET_B_RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids.pickle"

        result_file_names["FILE_TrainNewAE_KeepADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Mixed_300MinDistance_DATASET_A_RandomNumpyRandomNormalUniform_DATASET_B_RandomRandomRBF.pickle_2021-08-12_01.34_10ITERATIONS__fitNewAETrue_fitFalse_initNewADWINFalse_feedNewADWINFalse'
        result_file_names["FILE_TrainNewAE_InitializeADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Mixed_300MinDistance_DATASET_A_RandomNumpyRandomNormalUniform_DATASET_B_RandomRandomRBF.pickle_2021-08-12_10.52_10ITERATIONS__fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_TrainNewAE_InitializeAndFeedADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Mixed_300MinDistance_DATASET_A_RandomNumpyRandomNormalUniform_DATASET_B_RandomRandomRBF.pickle_2021-08-12_11.59_10ITERATIONS__fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINTrue'
        result_file_names["FILE_RetrainAE_KeepADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Mixed_300MinDistance_DATASET_A_RandomNumpyRandomNormalUniform_DATASET_B_RandomRandomRBF.pickle_2021-08-12_13.43_10ITERATIONS__fitNewAEFalse_fitTrue_initNewADWINFalse_feedNewADWINFalse'
        result_file_names["FILE_RetrainAE_InitializeADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Mixed_300MinDistance_DATASET_A_RandomNumpyRandomNormalUniform_DATASET_B_RandomRandomRBF.pickle_2021-08-12_14.21_10ITERATIONS__fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_RetrainAE_InitializeAndFeedADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Mixed_300MinDistance_DATASET_A_RandomNumpyRandomNormalUniform_DATASET_B_RandomRandomRBF.pickle_2021-08-12_15.59_10ITERATIONS__fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINTrue'

    elif experiment == '50DR_RandomRandomRBF_Min1DriftCentroid_0to1CS':
        path = "Generated_Streams/Drift_And_Classifier_Labels/"
        dataset = "RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids_300MinL_2000MaxL_2021-08-06_10.57.pickle"

        result_file_names["FILE_TrainNewAE_KeepADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids_300MinL_2000MaxL_2021-08-06_10.57.pickle_2021-08-09_01.15_10ITERATIONS__fitNewAETrue_fitFalse_initNewADWINFalse_feedNewADWINFalse'
        result_file_names["FILE_TrainNewAE_InitializeADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids_300MinL_2000MaxL_2021-08-06_10.57.pickle_2021-08-09_01.48_10ITERATIONS__fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_TrainNewAE_InitializeAndFeedADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids_300MinL_2000MaxL_2021-08-06_10.57.pickle_2021-08-09_03.04_10ITERATIONS__fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINTrue'
        result_file_names["FILE_RetrainAE_KeepADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids_300MinL_2000MaxL_2021-08-06_10.57.pickle_2021-08-09_05.01_10ITERATIONS__fitNewAEFalse_fitTrue_initNewADWINFalse_feedNewADWINFalse'
        result_file_names["FILE_RetrainAE_InitializeADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids_300MinL_2000MaxL_2021-08-06_10.57.pickle_2021-08-09_05.44_10ITERATIONS__fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_RetrainAE_InitializeAndFeedADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids_300MinL_2000MaxL_2021-08-06_10.57.pickle_2021-08-09_07.42_10ITERATIONS__fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINTrue'

    elif experiment == '50DR_RandomMNIST_(Fashion)MNIST_SortAllNumbers':
        path = "Generated_Streams/Image_Data_Drift_And_Classifier_Labels/"
        dataset = "RandomMNIST_and_FashionMNIST_SortAllNumbers19DR_2021-08-06_11.07.pickle"

        result_file_names["FILE_TrainNewAE_KeepADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomMNIST_and_FashionMNIST_SortAllNumbers19DR_2021-08-06_11.07.pickle_2021-08-09_11.34_10ITERATIONS__fitNewAETrue_fitFalse_initNewADWINFalse_feedNewADWINFalse'
        result_file_names["FILE_TrainNewAE_InitializeADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomMNIST_and_FashionMNIST_SortAllNumbers19DR_2021-08-06_11.07.pickle_2021-08-09_12.38_10ITERATIONS__fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_TrainNewAE_InitializeAndFeedADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomMNIST_and_FashionMNIST_SortAllNumbers19DR_2021-08-06_11.07.pickle_2021-08-11_04.16_1ITERATIONS__fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINTrue'
        result_file_names["FILE_RetrainAE_KeepADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomMNIST_and_FashionMNIST_SortAllNumbers19DR_2021-08-06_11.07.pickle_2021-08-11_05.35_1ITERATIONS__fitNewAEFalse_fitTrue_initNewADWINFalse_feedNewADWINFalse'
        result_file_names["FILE_RetrainAE_InitializeADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomMNIST_and_FashionMNIST_SortAllNumbers19DR_2021-08-06_11.07.pickle_2021-08-11_06.05_1ITERATIONS__fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_RetrainAE_InitializeAndFeedADWIN"] = 'SAW_Autoencoder_ADWIN_Training_RandomMNIST_and_FashionMNIST_SortAllNumbers19DR_2021-08-06_11.07.pickle_2021-08-11_09.40_1ITERATIONS__fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINTrue'

    # Experiments Time Complexity Analysis
    elif experiment == '10Dim':
        path = "Generated_Streams/Drift_Labels/"
        dataset = "Time_RandomNumpyRandomNormalUniform_10DR_10Dims_1MinDimBroken_300MinL_2000MaxL_2021-09-06_22.24.pickle"

        result_file_names["FILE_TrainNewAE_KeepADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_10Dims_1MinDimBroken_300MinL_2000MaxL_2021-09-06_22.24.pickle_2021-09-07_17.32_10ITERATIONS_0.5ENCODER_fitNewAETrue_fitFalse_initNewADWINFalse_feedNewADWINFalse'
        result_file_names["FILE_TrainNewAE_InitializeADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_10Dims_1MinDimBroken_300MinL_2000MaxL_2021-09-06_22.24.pickle_2021-09-07_17.37_10ITERATIONS_0.5ENCODER_fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_TrainNewAE_InitializeAndFeedADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_10Dims_1MinDimBroken_300MinL_2000MaxL_2021-09-06_22.24.pickle_2021-09-07_17.53_10ITERATIONS_0.5ENCODER_fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINTrue'
        result_file_names["FILE_RetrainAE_KeepADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_10Dims_1MinDimBroken_300MinL_2000MaxL_2021-09-06_22.24.pickle_2021-09-07_18.03_10ITERATIONS_0.5ENCODER_fitNewAEFalse_fitTrue_initNewADWINFalse_feedNewADWINFalse'
        result_file_names["FILE_RetrainAE_InitializeADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_10Dims_1MinDimBroken_300MinL_2000MaxL_2021-09-06_22.24.pickle_2021-09-07_18.10_10ITERATIONS_0.5ENCODER_fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_RetrainAE_InitializeAndFeedADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_10Dims_1MinDimBroken_300MinL_2000MaxL_2021-09-06_22.24.pickle_2021-09-07_18.25_10ITERATIONS_0.5ENCODER_fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINTrue'

    elif experiment == '50Dim':
        path = "Generated_Streams/Drift_Labels/"
        dataset = "Time_RandomNumpyRandomNormalUniform_10DR_50Dims_5MinDimBroken_300MinL_2000MaxL_2021-09-06_22.25.pickle"

        result_file_names["FILE_TrainNewAE_KeepADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_50Dims_5MinDimBroken_300MinL_2000MaxL_2021-09-06_22.25.pickle_2021-09-07_18.51_10ITERATIONS_0.5ENCODER_fitNewAETrue_fitFalse_initNewADWINFalse_feedNewADWINFalse'
        result_file_names["FILE_TrainNewAE_InitializeADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_50Dims_5MinDimBroken_300MinL_2000MaxL_2021-09-06_22.25.pickle_2021-09-07_19.02_10ITERATIONS_0.5ENCODER_fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_TrainNewAE_InitializeAndFeedADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_50Dims_5MinDimBroken_300MinL_2000MaxL_2021-09-06_22.25.pickle_2021-09-07_19.30_10ITERATIONS_0.5ENCODER_fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINTrue'
        result_file_names["FILE_RetrainAE_KeepADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_50Dims_5MinDimBroken_300MinL_2000MaxL_2021-09-06_22.25.pickle_2021-09-07_19.53_10ITERATIONS_0.5ENCODER_fitNewAEFalse_fitTrue_initNewADWINFalse_feedNewADWINFalse'
        result_file_names["FILE_RetrainAE_InitializeADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_50Dims_5MinDimBroken_300MinL_2000MaxL_2021-09-06_22.25.pickle_2021-09-07_20.03_10ITERATIONS_0.5ENCODER_fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_RetrainAE_InitializeAndFeedADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_50Dims_5MinDimBroken_300MinL_2000MaxL_2021-09-06_22.25.pickle_2021-09-07_20.29_10ITERATIONS_0.5ENCODER_fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINTrue'

    elif experiment == '100Dim':
        path = "Generated_Streams/Drift_Labels/"
        dataset = "Time_RandomNumpyRandomNormalUniform_10DR_100Dims_10MinDimBroken_300MinL_2000MaxL_2021-09-06_22.26.pickle"

        result_file_names["FILE_TrainNewAE_KeepADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_100Dims_10MinDimBroken_300MinL_2000MaxL_2021-09-06_22.26.pickle_2021-09-07_20.56_10ITERATIONS_0.5ENCODER_fitNewAETrue_fitFalse_initNewADWINFalse_feedNewADWINFalse'
        result_file_names["FILE_TrainNewAE_InitializeADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_100Dims_10MinDimBroken_300MinL_2000MaxL_2021-09-06_22.26.pickle_2021-09-07_21.13_10ITERATIONS_0.5ENCODER_fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_TrainNewAE_InitializeAndFeedADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_100Dims_10MinDimBroken_300MinL_2000MaxL_2021-09-06_22.26.pickle_2021-09-07_21.42_10ITERATIONS_0.5ENCODER_fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINTrue'
        result_file_names["FILE_RetrainAE_KeepADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_100Dims_10MinDimBroken_300MinL_2000MaxL_2021-09-06_22.26.pickle_2021-09-07_22.19_10ITERATIONS_0.5ENCODER_fitNewAEFalse_fitTrue_initNewADWINFalse_feedNewADWINFalse'
        result_file_names["FILE_RetrainAE_InitializeADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_100Dims_10MinDimBroken_300MinL_2000MaxL_2021-09-06_22.26.pickle_2021-09-07_22.37_10ITERATIONS_0.5ENCODER_fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_RetrainAE_InitializeAndFeedADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_100Dims_10MinDimBroken_300MinL_2000MaxL_2021-09-06_22.26.pickle_2021-09-07_23.00_10ITERATIONS_0.5ENCODER_fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINTrue'

    elif experiment == '500Dim':
        path = "Generated_Streams/Drift_Labels/"
        dataset = "Time_RandomNumpyRandomNormalUniform_10DR_500Dims_50MinDimBroken_300MinL_2000MaxL_2021-09-06_22.26.pickle"

        result_file_names["FILE_TrainNewAE_KeepADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_500Dims_50MinDimBroken_300MinL_2000MaxL_2021-09-06_22.26.pickle_2021-09-07_23.46_10ITERATIONS_0.5ENCODER_fitNewAETrue_fitFalse_initNewADWINFalse_feedNewADWINFalse'
        result_file_names["FILE_TrainNewAE_InitializeADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_500Dims_50MinDimBroken_300MinL_2000MaxL_2021-09-06_22.26.pickle_2021-09-08_00.06_10ITERATIONS_0.5ENCODER_fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_TrainNewAE_InitializeAndFeedADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_500Dims_50MinDimBroken_300MinL_2000MaxL_2021-09-06_22.26.pickle_2021-09-08_00.59_10ITERATIONS_0.5ENCODER_fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINTrue'
        result_file_names["FILE_RetrainAE_KeepADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_500Dims_50MinDimBroken_300MinL_2000MaxL_2021-09-06_22.26.pickle_2021-09-08_01.32_10ITERATIONS_0.5ENCODER_fitNewAEFalse_fitTrue_initNewADWINFalse_feedNewADWINFalse'
        result_file_names["FILE_RetrainAE_InitializeADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_500Dims_50MinDimBroken_300MinL_2000MaxL_2021-09-06_22.26.pickle_2021-09-08_01.52_10ITERATIONS_0.5ENCODER_fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_RetrainAE_InitializeAndFeedADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_500Dims_50MinDimBroken_300MinL_2000MaxL_2021-09-06_22.26.pickle_2021-09-08_02.20_10ITERATIONS_0.5ENCODER_fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINTrue'

    elif experiment == '1000Dim':
        path = "Generated_Streams/Drift_Labels/"
        dataset = "Time_RandomNumpyRandomNormalUniform_10DR_1000Dims_100MinDimBroken_300MinL_2000MaxL_2021-09-06_22.27.pickle"

        result_file_names["FILE_TrainNewAE_KeepADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_1000Dims_100MinDimBroken_300MinL_2000MaxL_2021-09-06_22.27.pickle_2021-09-08_03.21_10ITERATIONS_0.5ENCODER_fitNewAETrue_fitFalse_initNewADWINFalse_feedNewADWINFalse'
        result_file_names["FILE_TrainNewAE_InitializeADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_1000Dims_100MinDimBroken_300MinL_2000MaxL_2021-09-06_22.27.pickle_2021-09-08_04.06_10ITERATIONS_0.5ENCODER_fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_TrainNewAE_InitializeAndFeedADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_1000Dims_100MinDimBroken_300MinL_2000MaxL_2021-09-06_22.27.pickle_2021-09-08_05.24_10ITERATIONS_0.5ENCODER_fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINTrue'
        result_file_names["FILE_RetrainAE_KeepADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_1000Dims_100MinDimBroken_300MinL_2000MaxL_2021-09-06_22.27.pickle_2021-09-08_06.37_10ITERATIONS_0.5ENCODER_fitNewAEFalse_fitTrue_initNewADWINFalse_feedNewADWINFalse'
        result_file_names["FILE_RetrainAE_InitializeADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_1000Dims_100MinDimBroken_300MinL_2000MaxL_2021-09-06_22.27.pickle_2021-09-08_07.07_10ITERATIONS_0.5ENCODER_fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_RetrainAE_InitializeAndFeedADWIN"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_1000Dims_100MinDimBroken_300MinL_2000MaxL_2021-09-06_22.27.pickle_2021-09-08_08.11_10ITERATIONS_0.5ENCODER_fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINTrue'

    else:
        print("Error: Experiment identifier does not exist!")
        sys.exit()

    return path, dataset, result_file_names
