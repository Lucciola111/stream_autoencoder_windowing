import re
import pandas as pd
from EvaluationFunctions.LoadFrameworkDesignsFilenames import load_framework_designs_filenames
from EvaluationFunctions.Load_withIterations_Results import load_with_iterations_results
from Evaluation.Plot_DetectionRate_DetectionTime import plot_detectionrate_detectiontime
from Utility_Functions.BoldExtremeValues import bold_extreme_values


# 0.1 Read in file names of experiment
# experiment = 'onlyMeanDrift_VerySmallVariance_50DR_RandomNormal_Min1DimsBroken'
# experiment = 'onlyMeanDrift_SmallVariance_50DR_RandomNormal_Min1DimsBroken'
# experiment = 'onlyMeanDrift_HigherVariance_50DR_RandomNormal_Min1DimsBroken'
# experiment = 'onlyVarianceDrift_50DR_RandomNormal_Min1DimsBroken'
# experiment = '50DR_RandomNormal_Min100DimsBroken'
# experiment = '50DR_RandomNormal_Min1DimsBroken'

# experiment = '50DR_RandomRandomRBF_Min1DriftCentroid_0to1CS'
# experiment = '100DR_Merged_Min1Broken'
experiment = '50DR_RandomMNIST_(Fashion)MNIST_SortAllNumbers'

path, dataset, result_file_names = load_framework_designs_filenames(experiment=experiment)

file_names = ["FILE_TrainNewAE_KeepADWIN", "FILE_TrainNewAE_InitializeADWIN", "FILE_TrainNewAE_InitializeAndFeedADWIN",
              "FILE_RetrainAE_KeepADWIN", "FILE_RetrainAE_InitializeADWIN", "FILE_RetrainAE_InitializeAndFeedADWIN"]
experiment_names = ["NewAE_KeepADWIN", "NewAE_InitADWIN", "NewAE_InitFeedADWIN",
                    "RetrainAE_KeepADWIN", "RetrainAE_InitADWIN", "RetrainAE_InitFeedADWIN"]

all_performance_metrics = []
all_accuracy = []
all_time_per_example = []
for experiment_idx in range(len(file_names)):
    evaluation_results = load_with_iterations_results(
        file_name=result_file_names[file_names[experiment_idx]], result_folder='SAW_Autoencoder_ADWIN_Training')

    time_per_example = evaluation_results['Time per Example']
    all_time_per_example.append(time_per_example)
    if len(evaluation_results.columns) == 16 or len(evaluation_results.columns) == 2:
        accuracy = evaluation_results['Accuracy']
        all_accuracy.append(accuracy)
    if len(evaluation_results.columns) != 2:
        performance_metrics = evaluation_results.iloc[:, 0:14]
        performance_metrics.rename(index={0: experiment_names[experiment_idx]}, inplace=True)
        # performance_metrics.rename(index={"SAW": experiment_names[experiment_idx]}, inplace=True)
        all_performance_metrics.append(performance_metrics)


# Merge performance metrics of experiments
performance_metrics_merged = pd.concat(all_performance_metrics, axis=0)
# Write to csv file
performance_metrics_merged.to_csv('EvaluationFiles/Framework/EXPERIMENT_' + str(experiment) + '_EVALUATION_Framework.csv')

# Create plot
plot_file_name = 'DetectionRate_DetectionTime' + str(experiment) + '.pdf'
plot_detectionrate_detectiontime(
    performance_metrics_plot=[performance_metrics_merged], experiment_names_plot=[experiment],
    plot_title=False, plot_file_name=plot_file_name, latex_font=False)

# Generate LaTeX Table
for col in performance_metrics_merged.columns.get_level_values(0).unique():
    bolded_value = 'max'
    if col == "% Changes detected":
        bolded_value = 'near_100'
    if col == "Mean time until detection (instances)":
        bolded_value = 'min'
    performance_metrics_merged[col] = bold_extreme_values(performance_metrics_merged[[col]], bolded_value=bolded_value)
print(re.sub(' +', ' ', performance_metrics_merged.to_latex(escape=False)))






