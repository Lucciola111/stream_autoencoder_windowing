import sys
import numpy as np
import pandas as pd
from EvaluationFunctions.LoadCompetitorsFilenames import load_competitors_filenames
from TrainingFunctions.LoadDataStream import load_data_stream
from EvaluationFunctions.LoadResults import load_results
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

path, dataset, result_file_names = load_competitors_filenames(experiment=experiment)

# 1. Read in Dataset to get Labels
# Load data stream
data_stream = load_data_stream(dataset=dataset, path=path, separate_train_test_file=False,
                               image_data=False, drift_labels_known=True,
                               proxy_evaluation=False)
n_train_val_data = 1000
data_stream, drift_labels = data_stream[:, :-1], data_stream[:, -1]
drift_labels = drift_labels[n_train_val_data:]

# 2. Read in Files and generate evaluation metrics
# 1) SAW
# Load results
evaluation_results_SAW_NewAE = load_results_SAW(
    file_name=result_file_names["FILE_SAW_NewAE"], result_folder='SAW_Autoencoder_ADWIN_Training')
# Calculate performance metrics
performance_metrics_SAW_NewAE = generate_performance_metrics(
    drift_decisions=evaluation_results_SAW_NewAE["drift_decisions"],
    drift_labels=evaluation_results_SAW_NewAE["drift_labels"], acceptance_levels=acceptance_levels,
    simple_metrics=simple_metrics, complex_metrics=complex_metrics, index_name='SAW (New AE)')

evaluation_results_SAW_RetrainAE = load_results_SAW(
    file_name=result_file_names["FILE_SAW_RetrainAE"], result_folder='SAW_Autoencoder_ADWIN_Training')
# Calculate performance metrics
performance_metrics_SAW_RetrainAE = generate_performance_metrics(
    drift_decisions=evaluation_results_SAW_NewAE["drift_decisions"],
    drift_labels=evaluation_results_SAW_NewAE["drift_labels"], acceptance_levels=acceptance_levels,
    simple_metrics=simple_metrics, complex_metrics=complex_metrics, index_name='SAW (Retrain AE)')

# 2) Multiple ADWINs (ADWIN-10)
# Load results
evaluation_results_ADWIN10 = load_results(
    file_name=result_file_names["FILE_Baseline_ADWIN10"], result_folder='Baseline_MultipleADWINS')
# Calculate performance metrics
performance_metrics_ADWIN10 = generate_performance_metrics(
        drift_decisions=evaluation_results_ADWIN10["drift_decisions"], drift_labels=evaluation_results_ADWIN10["drift_labels"],
        acceptance_levels=acceptance_levels, simple_metrics=simple_metrics, complex_metrics=complex_metrics, index_name='ADWIN-10')

# 3.) IBDD
evaluation_results_IBDD = load_results(
    file_name=result_file_names["FILE_Competitor_IBDD"], result_folder='Competitor_IBDD')
performance_metrics_IBDD = generate_performance_metrics(
    drift_decisions=evaluation_results_IBDD["drift_decisions"], drift_labels=evaluation_results_IBDD["drift_labels"],
    acceptance_levels=acceptance_levels, simple_metrics=simple_metrics, complex_metrics=complex_metrics, index_name='IBDD')

# 4.) D3
evaluation_results_D3 = load_results(
    file_name=result_file_names["FILE_Competitor_D3"], result_folder='Competitor_D3')
performance_metrics_D3 = generate_performance_metrics(
    drift_decisions=evaluation_results_D3["drift_decisions"], drift_labels=evaluation_results_D3["drift_labels"],
    acceptance_levels=acceptance_levels, simple_metrics=simple_metrics, complex_metrics=complex_metrics, index_name='D3')

# Pay attention about file content (output.txt)!!!
# 5.) PCA-CD (Qahtan)
# drift_decisions_PCA_CD = np.loadtxt('../X_BaselinesOtherPapers/Qahtan2015_Code_PCA-CD_C++/PCA-CD/ChangeDetectionEvaluation/output.txt')
drift_decisions_PCA_CD = np.loadtxt('../X_BaselinesOtherPapers/Qahtan2015_Code_PCA-CD_C++/PCA-CD/ChangeDetectionEvaluation/' + result_file_names["FILE_Competitor_PCA-CD"] + '.txt')
# Select only drifts after the start of the test data
drift_decisions_PCA_CD = drift_decisions_PCA_CD[drift_decisions_PCA_CD > n_train_val_data]
drift_labels_PCA_CD = drift_labels
performance_metrics_PCA_CD = generate_performance_metrics(
        drift_decisions=drift_decisions_PCA_CD, drift_labels=drift_labels_PCA_CD, acceptance_levels=acceptance_levels,
        simple_metrics=simple_metrics, complex_metrics=complex_metrics, index_name='PCA-CD')


# Check if the same amount of data was used so that it is comparable!
if not (len(evaluation_results_SAW_NewAE["drift_decisions"]) == len(evaluation_results_SAW_RetrainAE["drift_decisions"])
        == len(evaluation_results_ADWIN10["drift_decisions"]) == len(evaluation_results_IBDD["drift_decisions"])
        == len(evaluation_results_D3["drift_decisions"]) == len(drift_labels)):
    print("Error: Files do not have the same number of test data!")
    sys.exit()

# Merge tables
performance_metrics = pd.concat(
    [performance_metrics_SAW_NewAE, performance_metrics_SAW_RetrainAE, performance_metrics_ADWIN10,
     performance_metrics_IBDD, performance_metrics_D3, performance_metrics_PCA_CD],
    axis=0)

performance_metrics.to_csv('EvaluationFiles/Competitors/EXPERIMENT_' + str(experiment) + '_EVALUATION_Competitors.csv')

# Create plot
plot_detectionrate_detectiontime(
    performance_metrics_plot=[performance_metrics], experiment_names_plot=[experiment],
    plot_title=False, plot_file_name=False, latex_font=False)

# Generate LaTeX Table
for col in performance_metrics.columns.get_level_values(0).unique():
    bolded_value = 'max'
    if col == "% Changes detected":
        bolded_value = 'near_100'
    if col == "Mean time until detection (instances)":
        bolded_value = 'min'
    performance_metrics[col] = bold_extreme_values(performance_metrics[[col]], bolded_value=bolded_value)
print(performance_metrics.to_latex(escape=False))



