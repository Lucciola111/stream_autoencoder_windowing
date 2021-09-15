import sys


def load_evaluation_where_filenames(experiment):
    """

    Parameters
    ----------
    experiment: Identifier of experiment for evaluation

    Returns the path and name of the dataset and a list with the file names of the results
    -------

    """
    where_file_names = {}

    # Experiments NAE-IAW
    if experiment == '1Dim_Broken_NAE-IAW':
        where_file_names["FILE_NAME_re_all"] = "SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_1DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-31_13.25.pickle_2021-09-01_14.58_WHERE_ALL_fitNewAETrue_fitFalse_singleGDFalse_initNewADWINTrue_feedNewADWINFalse"
        where_file_names["FILE_NAME_re_old_pattern"] = "SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_1DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-31_13.25.pickle_2021-09-01_14.58_WHERE_OLDPATTERN_fitNewAETrue_fitFalse_singleGDFalse_initNewADWINTrue_feedNewADWINFalse"
        drift_dims = slice(40, 41)

    elif experiment == '10Dim_Broken_NAE-IAW':
        where_file_names["FILE_NAME_re_all"] = "SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_1DR_100Dims_10MinDimBroken_300MinL_2000MaxL_2021-08-31_13.26.pickle_2021-08-31_16.50_WHERE_ALL_fitNewAETrue_fitFalse_singleGDFalse_initNewADWINTrue_feedNewADWINFalse"
        where_file_names["FILE_NAME_re_old_pattern"] = "SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_1DR_100Dims_10MinDimBroken_300MinL_2000MaxL_2021-08-31_13.26.pickle_2021-08-31_16.50_WHERE_OLDPATTERN_fitNewAETrue_fitFalse_singleGDFalse_initNewADWINTrue_feedNewADWINFalse"
        drift_dims = slice(40, 50)

    elif experiment == '50Dim_Broken_NAE-IAW':
        where_file_names["FILE_NAME_re_all"] = "SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_1DR_100Dims_50MinDimBroken_300MinL_2000MaxL_2021-08-31_13.26.pickle_2021-08-31_17.03_WHERE_ALL_fitNewAETrue_fitFalse_singleGDFalse_initNewADWINTrue_feedNewADWINFalse"
        where_file_names["FILE_NAME_re_old_pattern"] = "SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_1DR_100Dims_50MinDimBroken_300MinL_2000MaxL_2021-08-31_13.26.pickle_2021-08-31_17.03_WHERE_OLDPATTERN_fitNewAETrue_fitFalse_singleGDFalse_initNewADWINTrue_feedNewADWINFalse"
        drift_dims = slice(40, 90)

    elif experiment == '50Dim_LongAETraining_Broken_NAE-IAW':
        where_file_names["FILE_NAME_re_all"] = "SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_1DR_100Dims_50MinDimBroken_300MinL_2000MaxL_2021-08-31_13.26.pickle_2021-09-01_15.26_WHERE_ALL_fitNewAETrue_fitFalse_singleGDFalse_initNewADWINTrue_feedNewADWINFalse"
        where_file_names["FILE_NAME_re_old_pattern"] = "SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_1DR_100Dims_50MinDimBroken_300MinL_2000MaxL_2021-08-31_13.26.pickle_2021-09-01_15.26_WHERE_OLDPATTERN_fitNewAETrue_fitFalse_singleGDFalse_initNewADWINTrue_feedNewADWINFalse"
        drift_dims = slice(40, 90)

    elif experiment == '50Dim_BigBatchTraining_Broken_NAE-IAW':
        where_file_names["FILE_NAME_re_all"] = "SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_1DR_100Dims_50MinDimBroken_300MinL_2000MaxL_2021-08-31_13.26.pickle_2021-09-01_15.40_WHERE_ALL_fitNewAETrue_fitFalse_singleGDFalse_initNewADWINTrue_feedNewADWINFalse"
        where_file_names["FILE_NAME_re_old_pattern"] = "SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_1DR_100Dims_50MinDimBroken_300MinL_2000MaxL_2021-08-31_13.26.pickle_2021-09-01_15.40_WHERE_OLDPATTERN_fitNewAETrue_fitFalse_singleGDFalse_initNewADWINTrue_feedNewADWINFalse"
        drift_dims = slice(40, 90)

    elif experiment == '50Dim_NewDesign_Broken_NAE-IAW':
        where_file_names["FILE_NAME_re_all"] = "SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_1DR_100Dims_50MinDimBroken_300MinL_2000MaxL_2021-08-31_13.26.pickle_2021-09-02_14.42_WHERE_ALL_fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse"
        where_file_names["FILE_NAME_re_old_pattern"] = "SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_1DR_100Dims_50MinDimBroken_300MinL_2000MaxL_2021-08-31_13.26.pickle_2021-09-02_14.42_WHERE_OLDPATTERN_fitNewAETrue_fitFalse_initNewADWINTrue_feedNewADWINFalse"
        drift_dims = slice(40, 90)

    elif experiment == '100Dim_Broken_NAE-IAW':
        where_file_names["FILE_NAME_re_all"] = "SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_1DR_100Dims_100MinDimBroken_300MinL_2000MaxL_2021-08-31_13.27.pickle_2021-08-31_16.00_WHERE_ALL_fitNewAETrue_fitFalse_singleGDFalse_initNewADWINTrue_feedNewADWINFalse"
        where_file_names["FILE_NAME_re_old_pattern"] = "SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_1DR_100Dims_100MinDimBroken_300MinL_2000MaxL_2021-08-31_13.27.pickle_2021-08-31_16.00_WHERE_OLDPATTERN_fitNewAETrue_fitFalse_singleGDFalse_initNewADWINTrue_feedNewADWINFalse"
        drift_dims = slice(0, 100)

    # Experiments RAE-IAW
    elif experiment == '1Dim_Broken_RAE-IAW':
        where_file_names["FILE_NAME_re_all"] = "SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_1DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-31_13.25.pickle_2021-08-31_13.37_WHERE_ALL_fitNewAEFalse_fitTrue_singleGDFalse_initNewADWINTrue_feedNewADWINFalse"
        where_file_names["FILE_NAME_re_old_pattern"] = "SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_1DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-31_13.25.pickle_2021-08-31_13.37_WHERE_OLDPATTERN_fitNewAEFalse_fitTrue_singleGDFalse_initNewADWINTrue_feedNewADWINFalse"
        drift_dims = slice(40, 41)

    elif experiment == '10Dim_Broken_RAE-IAW':
        where_file_names["FILE_NAME_re_all"] = "SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_1DR_100Dims_10MinDimBroken_300MinL_2000MaxL_2021-08-31_13.26.pickle_2021-08-31_13.55_WHERE_ALL_fitNewAEFalse_fitTrue_singleGDFalse_initNewADWINTrue_feedNewADWINFalse"
        where_file_names["FILE_NAME_re_old_pattern"] = "SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_1DR_100Dims_10MinDimBroken_300MinL_2000MaxL_2021-08-31_13.26.pickle_2021-08-31_13.55_WHERE_OLDPATTERN_fitNewAEFalse_fitTrue_singleGDFalse_initNewADWINTrue_feedNewADWINFalse"
        drift_dims = slice(40, 50)

    elif experiment == '50Dim_Broken_RAE-IAW':
        where_file_names["FILE_NAME_re_all"] = "SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_1DR_100Dims_50MinDimBroken_300MinL_2000MaxL_2021-08-31_13.26.pickle_2021-08-31_14.02_WHERE_ALL_fitNewAEFalse_fitTrue_singleGDFalse_initNewADWINTrue_feedNewADWINFalse"
        where_file_names["FILE_NAME_re_old_pattern"] = "SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_1DR_100Dims_50MinDimBroken_300MinL_2000MaxL_2021-08-31_13.26.pickle_2021-08-31_14.02_WHERE_OLDPATTERN_fitNewAEFalse_fitTrue_singleGDFalse_initNewADWINTrue_feedNewADWINFalse"
        drift_dims = slice(40, 90)

    elif experiment == '100Dim_Broken_RAE-IAW':
        where_file_names["FILE_NAME_re_all"] = "SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_1DR_100Dims_100MinDimBroken_300MinL_2000MaxL_2021-08-31_13.27.pickle_2021-08-31_15.50_WHERE_ALL_fitNewAEFalse_fitTrue_singleGDFalse_initNewADWINTrue_feedNewADWINFalse"
        where_file_names["FILE_NAME_re_old_pattern"] = "SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_1DR_100Dims_100MinDimBroken_300MinL_2000MaxL_2021-08-31_13.27.pickle_2021-08-31_15.50_WHERE_OLDPATTERN_fitNewAEFalse_fitTrue_singleGDFalse_initNewADWINTrue_feedNewADWINFalse"
        drift_dims = slice(0, 100)
    else:
        print("Error: Experiment identifier does not exist!")
        sys.exit()

    return where_file_names, drift_dims
