import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from EvaluationFunctions.LoadEvaluationWhereFileNames import load_evaluation_where_filenames
from EvaluationFunctions.LoadResultsSAW_WHERE import load_results_SAW_where
from Evaluation.Plot_DistributionDimensionWhere import plot_distribution_dimension_where
from Evaluation.Plot_pvalueOverTime import plot_pvalue_over_time
from Evaluation.Plot_DriftScorePerDimension import plot_drift_score_per_dimension

# 0. Read in file names of experiment
experiments = ['1Dim_Broken_NAE-IAW', '10Dim_Broken_NAE-IAW', '50Dim_Broken_NAE-IAW', '100Dim_Broken_NAE-IAW']
               # '1Dim_Broken_RAE-IAW', '10Dim_Broken_RAE-IAW', '50Dim_Broken_RAE-IAW', '100Dim_Broken_RAE-IAW']
# experiments = ['50Dim_BigBatchTraining_Broken_NAE-IAW', '50Dim_LongAETraining_Broken_NAE-IAW', '50Dim_NewDesign_Broken_NAE-IAW']

accuracies = []
for experiment in experiments:
    accuracies_experiment = {}

    where_file_names, drift_dims = load_evaluation_where_filenames(experiment=experiment)
    result_folder = "SAW_Autoencoder_ADWIN_Training"
    file_names = ["FILE_NAME_re_all", "FILE_NAME_re_old_pattern"]
    experiment_names = ["new pattern", "old pattern"]

    # 1. Read in Files
    for experiment_idx in range(len(file_names)):
        # 1.1 Prepare both errors per dimension
        # Load results
        errors_per_dim, errors_per_dim_after_drift = load_results_SAW_where(
            where_file_names=where_file_names, file_names=file_names, result_folder=result_folder)
        # Take only values of the patterns before the detected drift point
        instances_before_detected_drift = len(errors_per_dim) - len(errors_per_dim_after_drift)
        errors_per_dim_before_drift = errors_per_dim[(instances_before_detected_drift - 200)
                                                     :- len(errors_per_dim_after_drift)]
        # Take only values within 200 instances
        errors_per_dim_after_drift = errors_per_dim_after_drift[:200]
        # 1.2 Fill array and apply statistical test incrementally
        # Iterate through instances
        p_time_points = []
        drift_scores_time_points = []
        df_p_value = pd.DataFrame()
        # Start with at least two instances for statistical test
        for time_idx in range(1, errors_per_dim_after_drift.shape[0], 1):
            # Incremental
            incremental_errors_per_dim_after_drift = errors_per_dim_after_drift[:time_idx+1]
            # Iterate through dimensions
            p_dimensions = []
            drift_scores_dimensions = []
            for dim in range(errors_per_dim.shape[1]):
                sample_error_per_dim_before_drift = errors_per_dim_before_drift[:, dim]
                sample_error_per_dim_after_drift = incremental_errors_per_dim_after_drift[:, dim]
                # Calculate Mann-Whitney U Test statistic
                U1, p = mannwhitneyu(sample_error_per_dim_before_drift, sample_error_per_dim_after_drift)
                # Calculate Drift Score
                drift_score = abs(np.median(sample_error_per_dim_before_drift) - np.median(sample_error_per_dim_after_drift))
                p_dimensions.append(p)
                drift_scores_dimensions.append(drift_score)
            p_time_points.append(p_dimensions)
            drift_scores_time_points.append(drift_scores_dimensions)

            # Create column indicating drift dimensions
            values_drift_dim = ['Non-drift dimension'] * 100
            values_drift_dim[drift_dims] = ['Drift dimension'] * len(range(*drift_dims.indices(100000)))
            # Create column indication time
            values_time = [time_idx+1] * 100

            if (time_idx+1) % 10 == 0 or time_idx == 1:
                # Create data frame
                df_p_value_time_point = pd.DataFrame(p_dimensions, columns=['p-value'])
                df_p_value_time_point.insert(loc=0, column='Drift score', value=drift_scores_dimensions)
                df_p_value_time_point.insert(loc=0, column='Time', value=values_time)
                df_p_value_time_point.insert(loc=0, column='Dimension Type', value=values_drift_dim)

                # Add local data frame to global data frame
                df_p_value = df_p_value.append(df_p_value_time_point)

        # Create Drift score plot over time
        plot_file_name = 'Figure_5_DriftScore_Over_Time_WhereBeforeAfterDrift_' + str(experiment) + '.pdf'
        plot_pvalue_over_time(df_p_value=df_p_value, value="Drift score", max_ylim=False, log=False,
                              plot_file_name=plot_file_name, latex_font=True)
        # Create p-value plot over time
        plot_file_name = 'Figure_5_PValue_Over_Time_WhereBeforeAfterDrift_' + str(experiment) + '.pdf'
        log = False if experiment == "10Dim_Broken_NAE-IAW" else True
        plot_pvalue_over_time(df_p_value=df_p_value, max_ylim=False, log=log,
                              plot_file_name=plot_file_name, latex_font=True)

        # Create plot with drift scores
        df_drift_scores = pd.DataFrame(drift_scores_time_points[-1], columns=['Drift score'])
        df_drift_scores.insert(loc=0, column='Dimension', value=np.arange(100)+1)
        df_drift_scores.insert(loc=0, column='Dimension Type', value=values_drift_dim)
        plot_file_name = 'Figure_5_Median_DriftScore_PerDimension_WhereBeforeAfterDrift_' + str(experiment) + '.pdf'
        plot_drift_score_per_dimension(df_drift_scores=df_drift_scores,
                                       plot_file_name=plot_file_name, latex_font=True)

        # Plot distribution of single dimension
        dimensions = [40, 10]
        fig, axs = plt.subplots(1, len(dimensions), sharey=True, figsize=(12, 5))
        # max_ylim = 20
        max_ylim = False
        labels = ["Distribution before drift", "Distribution after drift"]
        plot_file_name = 'Figure_5_Distribution_Dimension_WhereBeforeAfterDrift_' + str(experiment) \
                         + '_MaxYLim' + str(max_ylim) + '_DimA' + str(dimensions[0]) + '_DimB' + str(dimensions[1]) + '.pdf'
        for idx in range(len(dimensions)):
            drift = True if values_drift_dim[dimensions[idx]] == 'Drift dimension' else False
            plot_distribution_dimension_where(data1=errors_per_dim_before_drift, data2=errors_per_dim_after_drift,
                                              dimension=dimensions[idx], drift=drift, labels=labels, axis=axs[idx],
                                              max_ylim=max_ylim, latex_font=True)

        if plot_file_name:
            plt.savefig("Plots/Where/" + str(plot_file_name))
        plt.show()
