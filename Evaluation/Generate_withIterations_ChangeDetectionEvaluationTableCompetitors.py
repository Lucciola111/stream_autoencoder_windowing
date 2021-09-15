import re
import pandas as pd
from EvaluationFunctions.LoadCompetitorsFilenames import load_competitors_filenames
from EvaluationFunctions.Load_withIterations_Results import load_with_iterations_results
from Evaluation.Plot_DetectionRate_DetectionTime import plot_detectionrate_detectiontime
from Utility_Functions.BoldExtremeValues import bold_extreme_values


# 0.1 Read in file names of experiment
experiments = ['onlyMeanDrift_VerySmallVariance_50DR_RandomNormal_Min1DimsBroken',
               'onlyMeanDrift_SmallVariance_50DR_RandomNormal_Min1DimsBroken',
               'onlyMeanDrift_HigherVariance_50DR_RandomNormal_Min1DimsBroken',
               'onlyVarianceDrift_50DR_RandomNormal_Min1DimsBroken',
               '50DR_RandomNormal_Min100DimsBroken',
               '50DR_RandomNormal_Min1DimsBroken',
               '100DR_Merged_Min1Broken',
               '50DR_RandomRandomRBF_Min1DriftCentroid_0to1CS',
               '50DR_RandomMNIST_(Fashion)MNIST_SortAllNumbers']

all_performance_metrics_merged = pd.DataFrame()
for experiment in experiments:
    path, dataset, result_file_names = load_competitors_filenames(experiment=experiment)

    file_names = ["FILE_SAW_NewAE", "FILE_SAW_RetrainAE",
                  "FILE_Baseline_ADWIN10", "FILE_Baseline_ADWIN10-initialized",
                  "FILE_Competitor_IBDD", "FILE_Competitor_D3", "FILE_Competitor_PCA-CD"]
    result_folders = ["SAW_Autoencoder_ADWIN_Training", "SAW_Autoencoder_ADWIN_Training",
                      "Baseline_MultipleADWINS", "Baseline_MultipleADWINS",
                      "Competitor_IBDD", "Competitor_D3", "Competitor_PCA-CD"]
    experiment_names = ["SAW (NAE-IAW)", "SAW (RAE-IAW)",
                        "ADWIN-10", "ADWIN-10i",
                        "IBDD", "D3", "PCA-CD"]

    all_performance_metrics = []
    all_accuracy = []
    all_time_per_example = []

    # 1. Read in Files and generate evaluation metrics
    # Load results
    all_performance_metrics = []
    all_accuracy = []
    all_time_per_example = []
    invalid_PCACD = None
    for experiment_idx in range(len(file_names)):
        if result_file_names[file_names[experiment_idx]] != '-':
            evaluation_results = load_with_iterations_results(
                file_name=result_file_names[file_names[experiment_idx]], result_folder=result_folders[experiment_idx])

            time_per_example = evaluation_results['Time per Example']
            all_time_per_example.append(time_per_example)
            if len(evaluation_results.columns) == 16 or len(evaluation_results.columns) == 2:
                accuracy = evaluation_results['Accuracy']
                all_accuracy.append(accuracy)
            if len(evaluation_results.columns) != 2:
                performance_metrics = evaluation_results.iloc[:, 0:14]
                performance_metrics.rename(index={0: experiment_names[experiment_idx]}, inplace=True)
                all_performance_metrics.append(performance_metrics)

        else:
            invalid_PCACD = pd.DataFrame('-', index=[experiment_names[experiment_idx]], columns=[
                "% Changes detected", "Mean time until detection (instances)",
                "Precision_60", "Precision_120", "Precision_180", "Precision_300",
                "Recall_60", "Recall_120", "Recall_180", "Recall_300",
                "F1Score_60", "F1Score_120", "F1Score_180", "F1Score_300"])

    # Merge performance metrics of competitors for the experiment
    performance_metrics_merged = pd.concat(all_performance_metrics, axis=0)
    # Write to csv file
    performance_metrics_merged.to_csv(
        'EvaluationFiles/Competitors/EXPERIMENT_' + str(experiment) + '_EVALUATION_Competitors.csv')
    all_performance_metrics_merged = all_performance_metrics_merged.append(performance_metrics_merged)
    # Create plot
    plot_detectionrate_detectiontime(
        performance_metrics_plot=[performance_metrics_merged], experiment_names_plot=[experiment],
        plot_title=False, plot_file_name=False, latex_font=False)

    # Generate LaTeX Table
    for col in performance_metrics_merged.columns.get_level_values(0).unique():
        bolded_value = 'max'
        if col == "% Changes detected":
            bolded_value = 'near_100'
        if col == "Mean time until detection (instances)":
            bolded_value = 'min'
        performance_metrics_merged[col] = bold_extreme_values(performance_metrics_merged[[col]], bolded_value=bolded_value)
    if invalid_PCACD is not None:
        performance_metrics_merged = performance_metrics_merged.append(invalid_PCACD)
    print(re.sub(' +', ' ', performance_metrics_merged.to_latex(escape=False)))

# Generate Table Median
medians = pd.DataFrame()
for experiment in experiment_names:
    median_experiment = pd.DataFrame(all_performance_metrics_merged.loc[experiment].median(), columns=[experiment]).T
    medians = medians.append(median_experiment)

# Generate LaTeX Table Median
for col in medians.columns.get_level_values(0).unique():
    bolded_value = 'max'
    if col == "% Changes detected":
        bolded_value = 'near_100'
    if col == "Mean time until detection (instances)":
        bolded_value = 'min'
    medians[col] = bold_extreme_values(medians[[col]], bolded_value=bolded_value)
print(re.sub(' +', ' ', medians.to_latex(escape=False)))

