def load_sensitivity_study_file_names(design):
    """

    Parameters
    ----------
    design: Name of design

    Returns experiment files names of a design
    -------

    """
    file_names = {}
    if design == 'NAE-IAW':
        path = '../Files_Results/Sensitivity_Study/NAE-IAW/'

        file_names['FILE_MeanDrift_var0.01'] = 'Sensitivity_Study_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.01_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-10_10.32.pickle_2021-08-24_12.30_10ITERATIONS_fitNewAETrue_fitFalse'
        file_names['FILE_MeanDrift_var0.05'] = 'Sensitivity_Study_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.05_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.42.pickle_2021-08-24_14.36_10ITERATIONS_fitNewAETrue_fitFalse'
        file_names['FILE_MeanDrift_var0.25'] = 'Sensitivity_Study_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.25_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.45.pickle_2021-08-24_17.24_10ITERATIONS_fitNewAETrue_fitFalse'
        file_names['FILE_VarianceDrift'] = 'Sensitivity_Study_RandomNumpyRandomNormalUniform_onlyVarianceDrift_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_11.15.pickle_2021-08-24_20.58_10ITERATIONS_fitNewAETrue_fitFalse'
        file_names['FILE_MeanVarianceDrift_all_broken'] = 'Sensitivity_Study_RandomNumpyRandomNormalUniform_50DR_100Dims_100MinDimBroken_300MinL_2000MaxL_2021-08-06_10.54.pickle_2021-08-24_13.34_10ITERATIONS_fitNewAETrue_fitFalse'
        file_names['FILE_MeanVarianceDrift'] = 'Sensitivity_Study_RandomNumpyRandomNormalUniform_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.53.pickle_2021-08-24_16.22_10ITERATIONS_fitNewAETrue_fitFalse'
        file_names['FILE_RandomRBF_Generator'] = 'Sensitivity_Study_RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids_300MinL_2000MaxL_2021-08-06_10.57.pickle_2021-08-24_13.20_10ITERATIONS_fitNewAETrue_fitFalse'
        file_names['FILE_MergedStream'] = 'Sensitivity_Study_Mixed_300MinDistance_DATASET_A_RandomNumpyRandomNormalUniform_DATASET_B_RandomRandomRBF.pickle_2021-08-24_13.08_10ITERATIONS_fitNewAETrue_fitFalse'
        file_names['FILE_FashionMNIST'] = 'Sensitivity_Study_RandomMNIST_and_FashionMNIST_SortAllNumbers19DR_2021-08-06_11.07.pickle_2021-08-25_12.07_10ITERATIONS_fitNewAETrue_fitFalse'

    if design == 'RAE-IAW':
        path = '../Files_Results/Sensitivity_Study/RAE-IAW/'

        file_names['FILE_MeanDrift_var0.01'] = 'Sensitivity_Study_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.01_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-10_10.32.pickle_2021-08-11_19.36_10ITERATIONS_fitNewAEFalse_fitTrue'
        file_names['FILE_MeanDrift_var0.05'] = 'Sensitivity_Study_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.05_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.42.pickle_2021-08-12_00.04_10ITERATIONS_fitNewAEFalse_fitTrue'
        file_names['FILE_MeanDrift_var0.25'] = 'Sensitivity_Study_RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.25_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.45.pickle_2021-08-12_14.18_10ITERATIONS_fitNewAEFalse_fitTrue'
        file_names['FILE_VarianceDrift'] = 'Sensitivity_Study_RandomNumpyRandomNormalUniform_onlyVarianceDrift_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_11.15.pickle_2021-08-12_18.04_10ITERATIONS_fitNewAEFalse_fitTrue'
        file_names['FILE_MeanVarianceDrift_all_broken'] = 'Sensitivity_Study_RandomNumpyRandomNormalUniform_50DR_100Dims_100MinDimBroken_300MinL_2000MaxL_2021-08-06_10.54.pickle_2021-08-12_20.47_10ITERATIONS_fitNewAEFalse_fitTrue'
        file_names['FILE_MeanVarianceDrift'] = 'Sensitivity_Study_RandomNumpyRandomNormalUniform_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.53.pickle_2021-08-13_00.30_10ITERATIONS_fitNewAEFalse_fitTrue'
        file_names['FILE_RandomRBF_Generator'] = 'Sensitivity_Study_RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids_300MinL_2000MaxL_2021-08-06_10.57.pickle_2021-08-12_19.37_10ITERATIONS_fitNewAEFalse_fitTrue'
        file_names['FILE_MergedStream'] = 'Sensitivity_Study_Mixed_300MinDistance_DATASET_A_RandomNumpyRandomNormalUniform_DATASET_B_RandomRandomRBF.pickle_2021-08-13_03.54_10ITERATIONS_fitNewAEFalse_fitTrue'
        file_names['FILE_FashionMNIST'] = 'Sensitivity_Study_RandomMNIST_and_FashionMNIST_SortAllNumbers19DR_2021-08-06_11.07.pickle_2021-08-12_22.40_10ITERATIONS_fitNewAEFalse_fitTrue'

    return path, file_names
