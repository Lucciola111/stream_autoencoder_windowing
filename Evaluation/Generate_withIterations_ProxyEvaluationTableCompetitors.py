import re
import numpy as np
import pandas as pd
from EvaluationFunctions.LoadCompetitorsFilenames import load_competitors_filenames
from EvaluationFunctions.Load_withIterations_Results import load_with_iterations_results
from Utility_Functions.BoldExtremeValues import bold_extreme_values

# 0.1 Initialize Settings for Evaluation

# 0.2 Read in file names of experiment
experiments = []
experiments.append('50DR_RandomRandomRBF_Min1DriftCentroid_0to1CS')
experiments.append('50DR_RandomMNIST_(Fashion)MNIST_SortAllNumbers')
experiments.append('Yoga')
experiments.append('StarLightCurves')
experiments.append('Heartbeats')

accuracies = []
for experiment in experiments:
    accuracies_experiment = {}
    path, dataset, result_file_names = load_competitors_filenames(experiment=experiment)

    file_names = ["FILE_SAW_NewAE", "FILE_SAW_RetrainAE", "FILE_Baseline_ADWIN10", "FILE_Baseline_ADWIN10-initialized",
                  "FILE_Competitor_IBDD", "FILE_Competitor_D3", "FILE_Competitor_PCA-CD", "FILE_Competitor_Baseline"]
    result_folders = ["SAW_Autoencoder_ADWIN_Training", "SAW_Autoencoder_ADWIN_Training", "Baseline_MultipleADWINS",
                      "Baseline_MultipleADWINS", "Competitor_IBDD", "Competitor_D3", "Competitor_PCA-CD", "Baseline_StaticClassifier"]
    experiment_names = ["SAW (NAE-IAW)", "SAW (RAE-IAW)", "ADWIN-10", "ADWIN-10i", "IBDD", "D3", "PCA-CD", "Baseline"]

    # 2. Read in Files and generate evaluation metrics
    for experiment_idx in range(len(file_names)):
        if result_file_names[file_names[experiment_idx]] != '-':
            evaluation_results = load_with_iterations_results(
                file_name=result_file_names[file_names[experiment_idx]], result_folder=result_folders[experiment_idx])

            if len(evaluation_results.columns) == 16 or len(evaluation_results.columns) == 2:
                accuracy = np.round(np.mean(evaluation_results['Accuracy']), 4)
                accuracies_experiment[experiment_names[experiment_idx]] = accuracy
        else:
            accuracies_experiment[experiment_names[experiment_idx]] = 0

    # Append accuracies of experiment to list of all experiments
    accuracies.append(accuracies_experiment)

# Prepare mean row
experiments.append("Median")
accuracies.append({})
# Create data frame
accuracies_table = pd.DataFrame(data=accuracies, index=[experiments])
# Calculate Mean
accuracies_table.loc['Median'] = [accuracies_table.median()]
accuracy.to_csv('EvaluationFiles/Competitors/EXPERIMENT_' + str(experiment) + '_EVALUATION_Competitors.csv')

# Generate LaTeX Table
for row in accuracies_table.index.get_level_values(0).unique():
    bolded_value = 'max'
    accuracies_table.loc[row] = [pd.Series(
        data=bold_extreme_values(accuracies_table.loc[[row]], col=False, bolded_value=bolded_value),
    )]
print(re.sub(' +', ' ', accuracies_table.to_latex(escape=False, float_format="%.2f")))



