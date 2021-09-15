import pandas as pd
from EvaluationFunctions.LoadFrameworkDesignsFilenames import load_framework_designs_filenames
from EvaluationFunctions.LoadResultsSAW import load_results_SAW
from EvaluationFunctions.GeneratePerformanceMetrics import generate_performance_metrics
from Evaluation.Plot_DetectionRate_DetectionTime import plot_detectionrate_detectiontime
from Utility_Functions.BoldExtremeValues import bold_extreme_values

# 0.1 Initialize Settings for Evaluation
# Define acceptance levels
acceptance_levels = [60, 120, 180, 300]
# Define table parameters
simple_metrics = False
complex_metrics = True

# 0.2 Read in file names of experiment
experiment = '50DR_RandomNormal_CompleteSet_AllDimsBroken'
# experiment = '50DR_RandomNormal_CompleteSet_Min10DimsBroken'
# experiment = '50DR_RandomNormal_CompleteSet_Min1DimsBroken'
# experiment = 'onlyMeanDrift_50DR_RandomNormal_CompleteSet_Min1DimsBroken'

# experiment = '50DR_RandomRandomRBF_CompleteSet_Min1DriftCentroid_0to1CS'
# experiment = '50DR_RandomHyperplane_CompleteSet_2MinDimsBroken'
# experiment = '50DR_RandomMNIST_VariateTwoSets'

path, dataset, result_file_names = load_framework_designs_filenames(experiment=experiment)


# 1. Read in Files
# Train new autoencoder
# 1) Keep ADWIN
# Load results
evaluation_results_1 = load_results_SAW(
    file_name=result_file_names["FILE_TrainNewAE_KeepADWIN"], result_folder='SAW_Autoencoder_ADWIN_Training')
# Calculate performance metrics
performance_metrics_1 = generate_performance_metrics(
    drift_decisions=evaluation_results_1["drift_decisions"], drift_labels=evaluation_results_1["drift_labels"],
    acceptance_levels=acceptance_levels, simple_metrics=simple_metrics, complex_metrics=complex_metrics,
    index_name='NewAE_KeepADWIN')

# 2) Initialize ADWIN
# Load results
evaluation_results_2 = load_results_SAW(
    file_name=result_file_names["FILE_TrainNewAE_InitializeADWIN"], result_folder='SAW_Autoencoder_ADWIN_Training')
# Calculate performance metrics
performance_metrics_2 = generate_performance_metrics(
    drift_decisions=evaluation_results_2["drift_decisions"], drift_labels=evaluation_results_2["drift_labels"],
    acceptance_levels=acceptance_levels, simple_metrics=simple_metrics, complex_metrics=complex_metrics,
    index_name='NewAE_InitADWIN')

# 3) Initialize and Feed ADWIN
# Load results
evaluation_results_3 = load_results_SAW(
    file_name=result_file_names["FILE_TrainNewAE_InitializeAndFeedADWIN"], result_folder='SAW_Autoencoder_ADWIN_Training')
# Calculate performance metrics
performance_metrics_3 = generate_performance_metrics(
    drift_decisions=evaluation_results_3["drift_decisions"], drift_labels=evaluation_results_3["drift_labels"],
    acceptance_levels=acceptance_levels, simple_metrics=simple_metrics, complex_metrics=complex_metrics,
    index_name='NewAE_InitFeedADWIN')


# Retrain autoencoder
# 4) Keep ADWIN
# Load results
evaluation_results_4 = load_results_SAW(
    file_name=result_file_names["FILE_RetrainAE_KeepADWIN"], result_folder='SAW_Autoencoder_ADWIN_Training')
# Calculate performance metrics
performance_metrics_4 = generate_performance_metrics(
    drift_decisions=evaluation_results_4["drift_decisions"], drift_labels=evaluation_results_4["drift_labels"],
    acceptance_levels=acceptance_levels, simple_metrics=simple_metrics, complex_metrics=complex_metrics,
    index_name='RetrainAE_KeepADWIN')

# 5) Initialize ADWIN
# Load results
evaluation_results_5 = load_results_SAW(
    file_name=result_file_names["FILE_RetrainAE_InitializeADWIN"], result_folder='SAW_Autoencoder_ADWIN_Training')
# Calculate performance metrics
performance_metrics_5 = generate_performance_metrics(
    drift_decisions=evaluation_results_5["drift_decisions"], drift_labels=evaluation_results_5["drift_labels"],
    acceptance_levels=acceptance_levels, simple_metrics=simple_metrics, complex_metrics=complex_metrics,
    index_name='RetrainAE_InitADWIN')

# 6) Initialize and Feed ADWIN
# Load results
evaluation_results_6 = load_results_SAW(
    file_name=result_file_names["FILE_RetrainAE_InitializeAndFeedADWIN"], result_folder='SAW_Autoencoder_ADWIN_Training')
# Calculate performance metrics
performance_metrics_6 = generate_performance_metrics(
    drift_decisions=evaluation_results_6["drift_decisions"], drift_labels=evaluation_results_6["drift_labels"],
    acceptance_levels=acceptance_levels, simple_metrics=simple_metrics, complex_metrics=complex_metrics,
    index_name='RetrainAE_InitFeedADWIN')


performance_metrics = pd.concat([performance_metrics_1, performance_metrics_2, performance_metrics_3,
                                 performance_metrics_4, performance_metrics_5, performance_metrics_6], axis=0)

performance_metrics.to_csv('EvaluationFiles/Framework/EXPERIMENT_' + str(experiment) + '_EVALUATION_Framework.csv')


# Create plot
plot_file_name = 'DetectionRate_DetectionTime' + str(experiment) + '.pdf'
plot_detectionrate_detectiontime(
    performance_metrics_plot=[performance_metrics], experiment_names_plot=[experiment],
    plot_title=False, plot_file_name=plot_file_name, latex_font=True)

# Generate LaTeX Table
for col in performance_metrics.columns.get_level_values(0).unique():
    bolded_value = 'max'
    if col == "% Changes detected":
        bolded_value = 'near_100'
    if col == "Mean time until detection (instances)":
        bolded_value = 'min'
    performance_metrics[col] = bold_extreme_values(performance_metrics[[col]], bolded_value=bolded_value)
print(performance_metrics.to_latex(escape=False))




