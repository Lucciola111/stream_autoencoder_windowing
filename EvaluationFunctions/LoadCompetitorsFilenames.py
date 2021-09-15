import sys


def load_competitors_filenames(experiment):
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

        result_file_names["FILE_SAW_NewAE"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.01_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-10_10.32.pickle_2021-08-25_13.01_10ITERATIONS_0.5ENCODER_fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_SAW_RetrainAE"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.01_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-10_10.32.pickle_2021-08-13_14.42_10ITERATIONS_0.5ENCODER_fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10"] = 'Baseline_MultipleADWINs_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.01_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-10_10.32.pickle_2021-08-12_13.56_10ITERATIONS__agreementRate0.1_reinitializeADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10-initialized"] = 'Baseline_MultipleADWINs_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.01_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-10_10.32.pickle_2021-08-12_01.13_10ITERATIONS__agreementRate0.1_reinitializeADWINTrue'
        result_file_names["FILE_Competitor_IBDD"] = 'Competitor_IBDD_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.01_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-10_10.32.pickle_2021-08-12_01.01_10ITERATIONS__windowSize1000_epsilon3'
        result_file_names["FILE_Competitor_D3"] = 'Competitor_D3_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.01_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-10_10.32.pickle_2021-08-14_01.35_10ITERATIONS_'
        result_file_names["FILE_Competitor_PCA-CD"] = 'Competitor_PCA-CD_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.01_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-10_10.32.pickle_2021-08-11_23.27_withoutITERATIONS_150windowSize'

    elif experiment == 'onlyMeanDrift_SmallVariance_50DR_RandomNormal_Min1DimsBroken':
        path = "Generated_Streams/Drift_Labels/"
        dataset = "SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.05_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.42.pickle"

        # result_file_names["FILE_SAW_NewAE_untuned"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.05_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.42.pickle_2021-08-13_02.22_10ITERATIONS__fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_SAW_NewAE"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.05_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.42.pickle_2021-08-25_13.26_10ITERATIONS_0.5ENCODER_fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_SAW_RetrainAE"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.05_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.42.pickle_2021-08-13_15.10_10ITERATIONS_0.5ENCODER_fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10"] = 'Baseline_MultipleADWINs_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.05_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.42.pickle_2021-08-09_01.32_10ITERATIONS__agreementRate0.1_reinitializeADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10-initialized"] = 'Baseline_MultipleADWINs_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.05_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.42.pickle_2021-08-12_13.53_10ITERATIONS__agreementRate0.1_reinitializeADWINTrue'
        result_file_names["FILE_Competitor_IBDD"] = 'Competitor_IBDD_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.05_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.42.pickle_2021-08-09_01.32_10ITERATIONS__windowSize1000_epsilon3'
        result_file_names["FILE_Competitor_D3"] = 'Competitor_D3_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.05_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.42.pickle_2021-08-09_20.09_10ITERATIONS_'
        result_file_names["FILE_Competitor_PCA-CD"] = 'Competitor_PCA-CD_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.05_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.42.pickle_2021-08-09_17.11_withoutITERATIONS_150windowSize'

    elif experiment == 'onlyMeanDrift_HigherVariance_50DR_RandomNormal_Min1DimsBroken':
        path = "Generated_Streams/Drift_Labels/"
        dataset = "SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.25_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.45.pickle"

        result_file_names["FILE_SAW_NewAE"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.25_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.45.pickle_2021-08-25_13.58_10ITERATIONS_0.5ENCODER_fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_SAW_RetrainAE"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.25_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.45.pickle_2021-08-13_15.59_10ITERATIONS_0.5ENCODER_fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10"] = 'Baseline_MultipleADWINs_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.25_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.45.pickle_2021-08-09_03.43_10ITERATIONS__agreementRate0.1_reinitializeADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10-initialized"] = 'Baseline_MultipleADWINs_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.25_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.45.pickle_2021-08-13_10.51_10ITERATIONS__agreementRate0.1_reinitializeADWINTrue'
        result_file_names["FILE_Competitor_IBDD"] = 'Competitor_IBDD_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.25_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.45.pickle_2021-08-09_03.45_10ITERATIONS__windowSize1000_epsilon3'
        result_file_names["FILE_Competitor_D3"] = 'Competitor_D3_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.25_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.45.pickle_2021-08-09_20.16_10ITERATIONS_'
        result_file_names["FILE_Competitor_PCA-CD"] = 'Competitor_PCA-CD_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.25_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.45.pickle_2021-08-09_17.13_withoutITERATIONS_150windowSize'

    elif experiment == 'onlyVarianceDrift_50DR_RandomNormal_Min1DimsBroken':
        path = "Generated_Streams/Drift_Labels/"
        dataset = "RandomNumpyRandomNormalUniform_onlyVarianceDrift_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_11.15.pickle"

        result_file_names["FILE_SAW_NewAE"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyVarianceDrift_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_11.15.pickle_2021-08-25_14.33_10ITERATIONS_0.5ENCODER_fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_SAW_RetrainAE"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_onlyVarianceDrift_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_11.15.pickle_2021-08-13_16.45_10ITERATIONS_0.5ENCODER_fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10"] = 'Baseline_MultipleADWINs_RandomNumpyRandomNormalUniform_onlyVarianceDrift_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_11.15.pickle_2021-08-09_11.41_10ITERATIONS__agreementRate0.1_reinitializeADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10-initialized"] = 'Baseline_MultipleADWINs_RandomNumpyRandomNormalUniform_onlyVarianceDrift_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_11.15.pickle_2021-08-12_16.11_10ITERATIONS__agreementRate0.1_reinitializeADWINTrue'
        result_file_names["FILE_Competitor_IBDD"] = 'Competitor_IBDD_RandomNumpyRandomNormalUniform_onlyVarianceDrift_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_11.15.pickle_2021-08-09_05.48_10ITERATIONS__windowSize1000_epsilon3'
        result_file_names["FILE_Competitor_D3"] = 'Competitor_D3_RandomNumpyRandomNormalUniform_onlyVarianceDrift_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_11.15.pickle_2021-08-09_20.30_10ITERATIONS_'
        result_file_names["FILE_Competitor_PCA-CD"] = 'Competitor_PCA-CD_RandomNumpyRandomNormalUniform_onlyVarianceDrift_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_11.15.pickle_2021-08-09_17.13_withoutITERATIONS_150windowSize'

    elif experiment == '50DR_RandomNormal_Min100DimsBroken':
        path = "Generated_Streams/Drift_Labels/"
        dataset = "RandomNumpyRandomNormalUniform_50DR_100Dims_100MinDimBroken_300MinL_2000MaxL_2021-08-06_10.54.pickle"

        result_file_names["FILE_SAW_NewAE"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_50DR_100Dims_100MinDimBroken_300MinL_2000MaxL_2021-08-06_10.54.pickle_2021-08-25_15.08_10ITERATIONS_0.5ENCODER_fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_SAW_RetrainAE"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_50DR_100Dims_100MinDimBroken_300MinL_2000MaxL_2021-08-06_10.54.pickle_2021-08-13_17.21_10ITERATIONS_0.5ENCODER_fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10"] = 'Baseline_MultipleADWINs_RandomNumpyRandomNormalUniform_50DR_100Dims_100MinDimBroken_300MinL_2000MaxL_2021-08-06_10.54.pickle_2021-08-09_13.45_10ITERATIONS__agreementRate0.1_reinitializeADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10-initialized"] = 'Baseline_MultipleADWINs_RandomNumpyRandomNormalUniform_50DR_100Dims_100MinDimBroken_300MinL_2000MaxL_2021-08-06_10.54.pickle_2021-08-12_18.10_10ITERATIONS__agreementRate0.1_reinitializeADWINTrue'
        result_file_names["FILE_Competitor_IBDD"] = 'Competitor_IBDD_RandomNumpyRandomNormalUniform_50DR_100Dims_100MinDimBroken_300MinL_2000MaxL_2021-08-06_10.54.pickle_2021-08-09_07.52_10ITERATIONS__windowSize1000_epsilon3'
        result_file_names["FILE_Competitor_D3"] = 'Competitor_D3_RandomNumpyRandomNormalUniform_50DR_100Dims_100MinDimBroken_300MinL_2000MaxL_2021-08-06_10.54.pickle_2021-08-09_20.48_10ITERATIONS_'
        result_file_names["FILE_Competitor_PCA-CD"] = 'Competitor_PCA-CD_RandomNumpyRandomNormalUniform_50DR_100Dims_100MinDimBroken_300MinL_2000MaxL_2021-08-06_10.54.pickle_2021-08-09_17.13_withoutITERATIONS_150windowSize'

    elif experiment == '50DR_RandomNormal_Min1DimsBroken':
        path = "Generated_Streams/Drift_Labels/"
        dataset = "RandomNumpyRandomNormalUniform_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.53.pickle"

        result_file_names["FILE_SAW_NewAE"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.53.pickle_2021-08-25_15.42_10ITERATIONS_0.5ENCODER_fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_SAW_RetrainAE"] = 'SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.53.pickle_2021-08-13_18.05_10ITERATIONS_0.5ENCODER_fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10"] = 'Baseline_MultipleADWINs_RandomNumpyRandomNormalUniform_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.53.pickle_2021-08-09_15.45_10ITERATIONS__agreementRate0.1_reinitializeADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10-initialized"] = 'Baseline_MultipleADWINs_RandomNumpyRandomNormalUniform_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.53.pickle_2021-08-12_19.57_10ITERATIONS__agreementRate0.1_reinitializeADWINTrue'
        result_file_names["FILE_Competitor_IBDD"] = 'Competitor_IBDD_RandomNumpyRandomNormalUniform_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.53.pickle_2021-08-09_09.46_10ITERATIONS__windowSize1000_epsilon3'
        result_file_names["FILE_Competitor_D3"] = 'Competitor_D3_RandomNumpyRandomNormalUniform_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.53.pickle_2021-08-09_21.03_10ITERATIONS_'
        result_file_names["FILE_Competitor_PCA-CD"] = 'Competitor_PCA-CD_RandomNumpyRandomNormalUniform_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.53.pickle_2021-08-09_17.14_withoutITERATIONS_150windowSize'

    # Experiments other datasets
    elif experiment == '100DR_Merged_Min1Broken':
        path = "Generated_Streams/Drift_Labels/"
        dataset = "Mixed_300MinDistance_DATASET_A_RandomNumpyRandomNormalUniform_50DR_100Dims_1MinDimBroken_DATASET_B_RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids.pickle"

        result_file_names["FILE_SAW_NewAE"] = 'SAW_Autoencoder_ADWIN_Training_Mixed_300MinDistance_DATASET_A_RandomNumpyRandomNormalUniform_DATASET_B_RandomRandomRBF.pickle_2021-08-25_13.07_10ITERATIONS_0.5ENCODER_fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_SAW_RetrainAE"] = 'SAW_Autoencoder_ADWIN_Training_Mixed_300MinDistance_DATASET_A_RandomNumpyRandomNormalUniform_DATASET_B_RandomRandomRBF.pickle_2021-08-13_18.43_10ITERATIONS_0.5ENCODER_fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10"] = 'Baseline_MultipleADWINs_Mixed_300MinDistance_DATASET_A_RandomNumpyRandomNormalUniform_DATASET_B_RandomRandomRBF.pickle_2021-08-12_17.34_10ITERATIONS__agreementRate0.1_reinitializeADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10-initialized"] = 'Baseline_MultipleADWINs_Mixed_300MinDistance_DATASET_A_RandomNumpyRandomNormalUniform_DATASET_B_RandomRandomRBF.pickle_2021-08-12_22.59_10ITERATIONS__agreementRate0.1_reinitializeADWINTrue'
        result_file_names["FILE_Competitor_IBDD"] = 'Competitor_IBDD_Mixed_300MinDistance_DATASET_A_RandomNumpyRandomNormalUniform_50DR_100Dims_1MinDimBroken_DATASET_B_RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids.pickle_2021-08-09_12.59_10ITERATIONS__windowSize1000_epsilon3'
        result_file_names["FILE_Competitor_D3"] = 'Competitor_D3_Mixed_300MinDistance_DATASET_A_RandomNumpyRandomNormalUniform_50DR_100Dims_1MinDimBroken_DATASET_B_RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids.pickle_2021-08-09_21.10_10ITERATIONS_'
        # Not working - skip
        result_file_names["FILE_Competitor_PCA-CD"] = '-'

    elif experiment == '50DR_RandomRandomRBF_Min1DriftCentroid_0to1CS':
        path = "Generated_Streams/Drift_And_Classifier_Labels/"
        dataset = "RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids_300MinL_2000MaxL_2021-08-06_10.57.pickle"

        result_file_names["FILE_SAW_NewAE"] = 'SAW_Autoencoder_ADWIN_Training_RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids_300MinL_2000MaxL_2021-08-06_10.57.pickle_2021-08-25_14.28_10ITERATIONS_0.5ENCODER_fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_SAW_RetrainAE"] = 'SAW_Autoencoder_ADWIN_Training_RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids_300MinL_2000MaxL_2021-08-06_10.57.pickle_2021-08-13_15.07_10ITERATIONS_0.5ENCODER_fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10"] = 'Baseline_MultipleADWINs_RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids_300MinL_2000MaxL_2021-08-06_10.57.pickle_2021-08-13_00.21_10ITERATIONS__agreementRate0.1_reinitializeADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10-initialized"] = 'Baseline_MultipleADWINs_RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids_300MinL_2000MaxL_2021-08-06_10.57.pickle_2021-08-13_11.41_10ITERATIONS__agreementRate0.1_reinitializeADWINTrue'
        result_file_names["FILE_Competitor_IBDD"] = 'Competitor_IBDD_RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids_300MinL_2000MaxL_2021-08-06_10.57.pickle_2021-08-09_17.01_10ITERATIONS__windowSize1000_epsilon3'
        result_file_names["FILE_Competitor_D3"] = 'Competitor_D3_RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids_300MinL_2000MaxL_2021-08-06_10.57.pickle_2021-08-10_14.09_10ITERATIONS_'
        result_file_names["FILE_Competitor_PCA-CD"] = 'Competitor_PCA-CD_RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids_300MinL_2000MaxL_2021-08-06_10.57.pickle_2021-08-09_17.27_withoutITERATIONS_150windowSize'
        result_file_names["FILE_Competitor_Baseline"] = 'Baseline_StaticClassifier_RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids_300MinL_2000MaxL_2021-08-06_10.57.pickle_2021-08-25_23.04_10ITERATIONS_'

    elif experiment == '50DR_RandomMNIST_(Fashion)MNIST_SortAllNumbers':
        path = "Generated_Streams/Image_Data_Drift_And_Classifier_Labels/"
        dataset = "RandomMNIST_and_FashionMNIST_SortAllNumbers19DR_2021-08-06_11.07.pickle"

        result_file_names["FILE_SAW_NewAE"] = 'SAW_Autoencoder_ADWIN_Training_RandomMNIST_and_FashionMNIST_SortAllNumbers19DR_2021-08-06_11.07.pickle_2021-08-25_15.38_10ITERATIONS_0.5ENCODER_fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_SAW_RetrainAE"] = 'SAW_Autoencoder_ADWIN_Training_RandomMNIST_and_FashionMNIST_SortAllNumbers19DR_2021-08-06_11.07.pickle_2021-08-14_12.11_10ITERATIONS_0.5ENCODER_fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10"] = 'Baseline_MultipleADWINs_RandomMNIST_and_FashionMNIST_SortAllNumbers19DR_2021-08-06_11.07.pickle_2021-08-27_02.24_10ITERATIONS__agreementRate0.1_reinitializeADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10-initialized"] = 'Baseline_MultipleADWINs_RandomMNIST_and_FashionMNIST_SortAllNumbers19DR_2021-08-06_11.07.pickle_2021-08-26_12.13_10ITERATIONS__agreementRate0.1_reinitializeADWINTrue'
        result_file_names["FILE_Competitor_IBDD"] = 'Competitor_IBDD_RandomMNIST_and_FashionMNIST_SortAllNumbers19DR_2021-08-06_11.07.pickle_2021-08-25_19.42_10ITERATIONS__windowSize1000_epsilon3'
        result_file_names["FILE_Competitor_D3"] = 'Competitor_D3_RandomMNIST_and_FashionMNIST_SortAllNumbers19DR_2021-08-06_11.07.pickle_2021-08-11_05.00_10ITERATIONS_'
        # Not working - skip
        result_file_names["FILE_Competitor_PCA-CD"] = '-'
        result_file_names["FILE_Competitor_Baseline"] = 'Baseline_StaticClassifier_RandomMNIST_and_FashionMNIST_SortAllNumbers19DR_2021-08-06_11.07.pickle_2021-08-25_23.04_10ITERATIONS_'

    # Experiments on real-world datasets
    elif experiment == 'Yoga':
        path = "IBDD_Datasets/benchmark_real/"
        # Take care with dataset!
        dataset = "Yoga"

        result_file_names["FILE_SAW_NewAE"] = 'SAW_Autoencoder_ADWIN_Training_Yoga_2021-08-25_15.56_10ITERATIONS_0.5ENCODER_fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_SAW_RetrainAE"] = 'SAW_Autoencoder_ADWIN_Training_Yoga_2021-08-13_15.22_10ITERATIONS_0.5ENCODER_fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10"] = 'Baseline_MultipleADWINs_Yoga_2021-08-13_12.09_10ITERATIONS__agreementRate0.1_reinitializeADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10-initialized"] = 'Baseline_MultipleADWINs_Yoga_2021-08-13_17.35_10ITERATIONS__agreementRate0.1_reinitializeADWINTrue'
        result_file_names["FILE_Competitor_IBDD"] = 'Competitor_IBDD_Yoga_2021-08-13_16.21_10ITERATIONS__windowSize300_epsilon3'
        result_file_names["FILE_Competitor_D3"] = 'Competitor_D3_Yoga_2021-08-14_09.33_10ITERATIONS_'
        # Not working - skip
        result_file_names["FILE_Competitor_PCA-CD"] = '-'
        result_file_names["FILE_Competitor_Baseline"] = 'Baseline_StaticClassifier_Yoga_2021-08-25_22.57_10ITERATIONS_'

    elif experiment == 'StarLightCurves':
        path = "IBDD_Datasets/benchmark_real/"
        # Take care with dataset!
        dataset = "StarLightCurves"

        result_file_names["FILE_SAW_NewAE"] = 'SAW_Autoencoder_ADWIN_Training_StarLightCurves_2021-08-25_16.00_10ITERATIONS_0.5ENCODER_fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_SAW_RetrainAE"] = 'SAW_Autoencoder_ADWIN_Training_StarLightCurves_2021-08-13_15.25_10ITERATIONS_0.5ENCODER_fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10"] = 'Baseline_MultipleADWINs_StarLightCurves_2021-08-13_00.32_10ITERATIONS__agreementRate0.1_reinitializeADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10-initialized"] = 'Baseline_MultipleADWINs_StarLightCurves_2021-08-13_00.39_10ITERATIONS__agreementRate0.1_reinitializeADWINTrue'
        result_file_names["FILE_Competitor_IBDD"] = 'Competitor_IBDD_StarLightCurves_2021-08-13_12.34_10ITERATIONS__windowSize1000_epsilon3'
        result_file_names["FILE_Competitor_D3"] = 'Competitor_D3_StarLightCurves_2021-08-14_10.20_10ITERATIONS_'
        # Not working - skip
        result_file_names["FILE_Competitor_PCA-CD"] = '-'
        result_file_names["FILE_Competitor_Baseline"] = 'Baseline_StaticClassifier_StarLightCurves_2021-08-25_22.59_10ITERATIONS_'

    elif experiment == 'Heartbeats':
        path = "IBDD_Datasets/benchmark_real/"
        # Take care with dataset!
        dataset = "Heartbeats"

        result_file_names["FILE_SAW_NewAE"] = 'SAW_Autoencoder_ADWIN_Training_Heartbeats_2021-08-25_16.41_10ITERATIONS_0.5ENCODER_fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_SAW_RetrainAE"] = 'SAW_Autoencoder_ADWIN_Training_Heartbeats_2021-08-13_16.06_10ITERATIONS_0.5ENCODER_fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10"] = 'Baseline_MultipleADWINs_Heartbeats_2021-08-13_20.48_10ITERATIONS__agreementRate0.1_reinitializeADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10-initialized"] = 'Baseline_MultipleADWINs_Heartbeats_2021-08-14_00.53_10ITERATIONS__agreementRate0.1_reinitializeADWINTrue'
        result_file_names["FILE_Competitor_IBDD"] = 'Competitor_IBDD_Heartbeats_2021-08-13_22.27_10ITERATIONS__windowSize500_epsilon3'
        result_file_names["FILE_Competitor_D3"] = 'Competitor_D3_Heartbeats_2021-08-23_19.47_10ITERATIONS_'
        # Not working - skip
        result_file_names["FILE_Competitor_PCA-CD"] = '-'
        result_file_names["FILE_Competitor_Baseline"] = 'Baseline_StaticClassifier_Heartbeats_2021-08-25_23.00_10ITERATIONS_'

    # Experiments Time Complexity Analysis
    elif experiment == '10Dim':
        path = "Generated_Streams/Drift_Labels/"
        dataset = "Time_RandomNumpyRandomNormalUniform_10DR_10Dims_1MinDimBroken_300MinL_2000MaxL_2021-09-06_22.24.pickle"

        result_file_names["FILE_SAW_NewAE"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_10Dims_1MinDimBroken_300MinL_2000MaxL_2021-09-06_22.24.pickle_2021-09-07_17.37_10ITERATIONS_0.5ENCODER_fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_SAW_RetrainAE"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_10Dims_1MinDimBroken_300MinL_2000MaxL_2021-09-06_22.24.pickle_2021-09-07_18.10_10ITERATIONS_0.5ENCODER_fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10"] = 'Baseline_MultipleADWINs_Time_RandomNumpyRandomNormalUniform_10DR_10Dims_1MinDimBroken_300MinL_2000MaxL_2021-09-06_22.24.pickle_2021-09-11_13.31_10ITERATIONS__agreementRate0.1_reinitializeADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10-initialized"] = 'Baseline_MultipleADWINs_Time_RandomNumpyRandomNormalUniform_10DR_10Dims_1MinDimBroken_300MinL_2000MaxL_2021-09-06_22.24.pickle_2021-09-10_21.40_10ITERATIONS__agreementRate0.1_reinitializeADWINTrue'
        result_file_names["FILE_Competitor_IBDD"] = 'Competitor_IBDD_Time_RandomNumpyRandomNormalUniform_10DR_10Dims_1MinDimBroken_300MinL_2000MaxL_2021-09-06_22.24.pickle_2021-09-11_23.41_10ITERATIONS__windowSize1000_epsilon3'
        result_file_names["FILE_Competitor_D3"] = 'Competitor_D3_Time_RandomNumpyRandomNormalUniform_10DR_10Dims_1MinDimBroken_300MinL_2000MaxL_2021-09-06_22.24.pickle_2021-09-11_12.02_10ITERATIONS_'
        # Not working - skip
        result_file_names["FILE_Competitor_PCA-CD"] = '-'

    elif experiment == '50Dim':
        path = "Generated_Streams/Drift_Labels/"
        dataset = "Time_RandomNumpyRandomNormalUniform_10DR_50Dims_5MinDimBroken_300MinL_2000MaxL_2021-09-06_22.25.pickle"

        result_file_names["FILE_SAW_NewAE"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_50Dims_5MinDimBroken_300MinL_2000MaxL_2021-09-06_22.25.pickle_2021-09-07_19.02_10ITERATIONS_0.5ENCODER_fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_SAW_RetrainAE"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_50Dims_5MinDimBroken_300MinL_2000MaxL_2021-09-06_22.25.pickle_2021-09-07_20.03_10ITERATIONS_0.5ENCODER_fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10"] = 'Baseline_MultipleADWINs_Time_RandomNumpyRandomNormalUniform_10DR_50Dims_5MinDimBroken_300MinL_2000MaxL_2021-09-06_22.25.pickle_2021-09-11_13.44_10ITERATIONS__agreementRate0.1_reinitializeADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10-initialized"] = 'Baseline_MultipleADWINs_Time_RandomNumpyRandomNormalUniform_10DR_50Dims_5MinDimBroken_300MinL_2000MaxL_2021-09-06_22.25.pickle_2021-09-10_21.50_10ITERATIONS__agreementRate0.1_reinitializeADWINTrue'
        result_file_names["FILE_Competitor_IBDD"] = 'Competitor_IBDD_Time_RandomNumpyRandomNormalUniform_10DR_50Dims_5MinDimBroken_300MinL_2000MaxL_2021-09-06_22.25.pickle_2021-09-12_00.12_10ITERATIONS__windowSize1000_epsilon3'
        result_file_names["FILE_Competitor_D3"] = 'Competitor_D3_Time_RandomNumpyRandomNormalUniform_10DR_50Dims_5MinDimBroken_300MinL_2000MaxL_2021-09-06_22.25.pickle_2021-09-11_12.04_10ITERATIONS_'
        # Not working - skip
        result_file_names["FILE_Competitor_PCA-CD"] = '-'

    elif experiment == '100Dim':
        path = "Generated_Streams/Drift_Labels/"
        dataset = "Time_RandomNumpyRandomNormalUniform_10DR_100Dims_10MinDimBroken_300MinL_2000MaxL_2021-09-06_22.26.pickle"

        result_file_names["FILE_SAW_NewAE"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_100Dims_10MinDimBroken_300MinL_2000MaxL_2021-09-06_22.26.pickle_2021-09-07_21.13_10ITERATIONS_0.5ENCODER_fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_SAW_RetrainAE"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_100Dims_10MinDimBroken_300MinL_2000MaxL_2021-09-06_22.26.pickle_2021-09-07_22.37_10ITERATIONS_0.5ENCODER_fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10"] = 'Baseline_MultipleADWINs_Time_RandomNumpyRandomNormalUniform_10DR_100Dims_10MinDimBroken_300MinL_2000MaxL_2021-09-06_22.26.pickle_2021-09-11_14.10_10ITERATIONS__agreementRate0.1_reinitializeADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10-initialized"] = 'Baseline_MultipleADWINs_Time_RandomNumpyRandomNormalUniform_10DR_100Dims_10MinDimBroken_300MinL_2000MaxL_2021-09-06_22.26.pickle_2021-09-10_22.12_10ITERATIONS__agreementRate0.1_reinitializeADWINTrue'
        result_file_names["FILE_Competitor_IBDD"] = 'Competitor_IBDD_Time_RandomNumpyRandomNormalUniform_10DR_100Dims_10MinDimBroken_300MinL_2000MaxL_2021-09-06_22.26.pickle_2021-09-12_01.01_10ITERATIONS__windowSize1000_epsilon3'
        result_file_names["FILE_Competitor_D3"] = 'Competitor_D3_Time_RandomNumpyRandomNormalUniform_10DR_100Dims_10MinDimBroken_300MinL_2000MaxL_2021-09-06_22.26.pickle_2021-09-11_12.05_10ITERATIONS_'
        # Not working - skip
        result_file_names["FILE_Competitor_PCA-CD"] = '-'

    elif experiment == '500Dim':
        path = "Generated_Streams/Drift_Labels/"
        dataset = "Time_RandomNumpyRandomNormalUniform_10DR_500Dims_50MinDimBroken_300MinL_2000MaxL_2021-09-06_22.26.pickle"

        result_file_names["FILE_SAW_NewAE"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_500Dims_50MinDimBroken_300MinL_2000MaxL_2021-09-06_22.26.pickle_2021-09-08_00.06_10ITERATIONS_0.5ENCODER_fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_SAW_RetrainAE"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_500Dims_50MinDimBroken_300MinL_2000MaxL_2021-09-06_22.26.pickle_2021-09-08_01.52_10ITERATIONS_0.5ENCODER_fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10"] = 'Baseline_MultipleADWINs_Time_RandomNumpyRandomNormalUniform_10DR_500Dims_50MinDimBroken_300MinL_2000MaxL_2021-09-06_22.26.pickle_2021-09-11_15.39_10ITERATIONS__agreementRate0.1_reinitializeADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10-initialized"] = 'Baseline_MultipleADWINs_Time_RandomNumpyRandomNormalUniform_10DR_500Dims_50MinDimBroken_300MinL_2000MaxL_2021-09-06_22.26.pickle_2021-09-10_23.28_10ITERATIONS__agreementRate0.1_reinitializeADWINTrue'
        result_file_names["FILE_Competitor_IBDD"] = 'Competitor_IBDD_Time_RandomNumpyRandomNormalUniform_10DR_500Dims_50MinDimBroken_300MinL_2000MaxL_2021-09-06_22.26.pickle_2021-09-12_03.41_10ITERATIONS__windowSize1000_epsilon3'
        result_file_names["FILE_Competitor_D3"] = 'Competitor_D3_Time_RandomNumpyRandomNormalUniform_10DR_500Dims_50MinDimBroken_300MinL_2000MaxL_2021-09-06_22.26.pickle_2021-09-11_12.07_10ITERATIONS_'
        # Not working - skip
        result_file_names["FILE_Competitor_PCA-CD"] = '-'

    elif experiment == '1000Dim':
        path = "Generated_Streams/Drift_Labels/"
        dataset = "Time_RandomNumpyRandomNormalUniform_10DR_1000Dims_100MinDimBroken_300MinL_2000MaxL_2021-09-06_22.27.pickle"

        result_file_names["FILE_SAW_NewAE"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_1000Dims_100MinDimBroken_300MinL_2000MaxL_2021-09-06_22.27.pickle_2021-09-08_04.06_10ITERATIONS_0.5ENCODER_fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_SAW_RetrainAE"] = 'SAW_Autoencoder_ADWIN_Training_Time_RandomNumpyRandomNormalUniform_10DR_1000Dims_100MinDimBroken_300MinL_2000MaxL_2021-09-06_22.27.pickle_2021-09-08_07.07_10ITERATIONS_0.5ENCODER_fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10"] = 'Baseline_MultipleADWINs_Time_RandomNumpyRandomNormalUniform_10DR_1000Dims_100MinDimBroken_300MinL_2000MaxL_2021-09-06_22.27.pickle_2021-09-11_19.49_10ITERATIONS__agreementRate0.1_reinitializeADWINFalse'
        result_file_names["FILE_Baseline_ADWIN10-initialized"] = 'Baseline_MultipleADWINs_Time_RandomNumpyRandomNormalUniform_10DR_1000Dims_100MinDimBroken_300MinL_2000MaxL_2021-09-06_22.27.pickle_2021-09-11_03.02_10ITERATIONS__agreementRate0.1_reinitializeADWINTrue'
        result_file_names["FILE_Competitor_IBDD"] = 'Competitor_IBDD_Time_RandomNumpyRandomNormalUniform_10DR_1000Dims_100MinDimBroken_300MinL_2000MaxL_2021-09-06_22.27.pickle_2021-09-12_10.45_10ITERATIONS__windowSize1000_epsilon3'
        result_file_names["FILE_Competitor_D3"] = 'Competitor_D3_Time_RandomNumpyRandomNormalUniform_10DR_1000Dims_100MinDimBroken_300MinL_2000MaxL_2021-09-06_22.27.pickle_2021-09-11_12.09_10ITERATIONS_'
        # Not working - skip
        result_file_names["FILE_Competitor_PCA-CD"] = '-'

    else:
        print("Error: Experiment identifier does not exist!")
        sys.exit()

    return path, dataset, result_file_names
