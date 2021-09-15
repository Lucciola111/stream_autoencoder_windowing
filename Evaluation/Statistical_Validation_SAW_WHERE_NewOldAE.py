
import pandas as pd
from scipy.stats import mannwhitneyu
from EvaluationFunctions.LoadEvaluationWhereFileNames import load_evaluation_where_filenames
from EvaluationFunctions.LoadResultsSAW_WHERE import load_results_SAW_where
from Evaluation.Plot_DistributionDimensionWhere import plot_distribution_dimension_where
from Evaluation.Plot_pvalueOverTime import plot_pvalue_over_time

# 0. Read in file names of experiment
experiments = ['1Dim_Broken_NAE-IAW', '10Dim_Broken_NAE-IAW', '50Dim_Broken_NAE-IAW', '100Dim_Broken_NAE-IAW']
experiments = ['1Dim_Broken_RAE-IAW', '10Dim_Broken_RAE-IAW', '50Dim_Broken_RAE-IAW', '100Dim_Broken_RAE-IAW']
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
        errors_per_dim, errors_per_dim_oldAE = load_results_SAW_where(
            where_file_names=where_file_names, file_names=file_names, result_folder=result_folder)
        # Take only new patterns as of the detected drift point
        errors_per_dim_cutted = errors_per_dim[
                                          (len(errors_per_dim) - len(errors_per_dim_oldAE)):]
        # Take only values within 300 instances
        errors_per_dim = errors_per_dim_cutted[:200]
        errors_per_dim_oldAE = errors_per_dim_oldAE[:200]
        # 1.2 Fill array and apply statistical test incrementally
        # Iterate through instances
        p_time_points = []
        df_p_value = pd.DataFrame()
        # Start with at least two instances for statistical test
        for time_idx in range(1, errors_per_dim.shape[0], 1):
            incremental_errors_per_dim = errors_per_dim[:time_idx+1]
            incremental_errors_per_dim_oldAE = errors_per_dim_oldAE[:time_idx+1]
            # Iterate through dimensions
            p_dimensions = []
            for dim in range(errors_per_dim.shape[1]):
                sample_error_per_dim = incremental_errors_per_dim[:, dim]
                sample_error_per_dim_oldAE = incremental_errors_per_dim_oldAE[:, dim]
                # Calculate Mann-Whitney U Test statistic
                U1, p = mannwhitneyu(sample_error_per_dim, sample_error_per_dim_oldAE)
                p_dimensions.append(p)
            p_time_points.append(p_dimensions)

            # Create column indicating drift dimensions
            values_drift_dim = ['Non-drift dimension'] * 100
            values_drift_dim[drift_dims] = ['Drift dimension'] * len(range(*drift_dims.indices(100000)))
            # Create column indication time
            values_time = [time_idx+1] * 100

            if (time_idx+1) % 10 == 0 or time_idx == 1:
                # Create data frame
                df_p_value_time_point = pd.DataFrame(p_dimensions, columns=['p-value'])
                df_p_value_time_point.insert(loc=0, column='Time', value=values_time)
                df_p_value_time_point.insert(loc=0, column='Dimension Type', value=values_drift_dim)

                # Add local data frame to global data frame
                df_p_value = df_p_value.append(df_p_value_time_point)

        # Create categorical plot over time
        max_ylim = False
        plot_file_name = 'Figure_5_PValue_Over_Time_WhereNewOldAE_' + str(experiment) + '_MaxYLim' + str(max_ylim) + '.pdf'
        plot_pvalue_over_time(df_p_value=df_p_value, max_ylim=max_ylim,
                              plot_file_name=plot_file_name, latex_font=True)

        # Plot distribution of single dimension
        dimensions = [10, 40]
        # max_ylim = 20
        max_ylim = False
        labels = ["New AE", "Old AE"]
        for dimension in dimensions:
            plot_file_name = 'Figure_5_Distribution_Dimension_WhereNewOldAE_' + str(experiment) + '_MaxYLim' + str(max_ylim) \
                             + '_Dim' + str(dimension) + '.pdf'
            drift = True if values_drift_dim[dimension] == 'Drift dimension' else False
            plot_distribution_dimension_where(data1=errors_per_dim, data2=errors_per_dim_oldAE,
                                              dimension=dimension, drift=drift, labels=labels, max_ylim=max_ylim,
                                              plot_file_name=plot_file_name, latex_font=True)

