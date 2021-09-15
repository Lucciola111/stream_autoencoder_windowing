import sys
import numpy as np
import pandas as pd
from EvaluationFunctions.LoadCompetitorsFilenames import load_competitors_filenames
from EvaluationFunctions.LoadResults import load_results
from EvaluationFunctions.LoadResultsSAW import load_results_SAW
from Utility_Functions.BoldExtremeValues import bold_extreme_values

# 0.1 Initialize Settings for Evaluation

# 0.2 Read in file names of experiment
experiments = []
experiments.append('50DR_RandomRandomRBF_CompleteSet_Min1DriftCentroid_0to1CS')
experiments.append('50DR_RandomMNIST_VariateTwoSets')

accuracies = []
for experiment in experiments:
    accuracies_experiment = {}
    path, dataset, result_file_names = load_competitors_filenames(experiment=experiment)

    # 2. Read in Files and generate evaluation metrics
    # 1) SAW
    # SAW Version 1
    # Load results
    evaluation_results_SAW_NewAE = load_results_SAW(
        file_name=result_file_names["FILE_SAW_NewAE"], result_folder='SAW_Autoencoder_ADWIN_Training')
    # Calculate accuracy
    accuracy_SAW_NewAE = np.round(np.mean(evaluation_results_SAW_NewAE["acc_vector"]) * 100, 4)
    # Append accuracy for that experiment to list
    accuracies_experiment["SAW_NewAE"] = accuracy_SAW_NewAE
    # SAW Version 2
    # Load results
    evaluation_results_SAW_RetrainAE = load_results_SAW(
        file_name=result_file_names["FILE_SAW_RetrainAE"], result_folder='SAW_Autoencoder_ADWIN_Training')
    # Calculate mean accuracy of classifier
    accuracy_SAW_RetrainAE = np.round(np.mean(evaluation_results_SAW_RetrainAE["acc_vector"]) * 100, 4)
    # Append accuracy for that experiment to list
    accuracies_experiment["SAW_RetrainAE"] = accuracy_SAW_RetrainAE

    # 2) Multiple ADWINs - ADWIN-10
    # Load results
    evaluation_results_ADWIN10 = load_results(
        file_name=result_file_names["FILE_Baseline_ADWIN10"], result_folder='Baseline_MultipleADWINS')
    # Calculate mean accuracy of classifier
    accuracy_ADWIN10 = np.round(np.mean(evaluation_results_ADWIN10["acc_vector"]) * 100, 4)
    # Append accuracy for that experiment to list
    accuracies_experiment["ADWIN-10"] = accuracy_ADWIN10

    # 3.) IBDD
    evaluation_results_IBDD = load_results(
        file_name=result_file_names["FILE_Competitor_IBDD"], result_folder='Competitor_IBDD')
    # Calculate mean accuracy of classifier
    accuracy_IBDD = np.round(np.mean(evaluation_results_IBDD["acc_vector"]) * 100, 4)
    # Append accuracy for that experiment to list
    accuracies_experiment["IBDD"] = accuracy_IBDD

    # 4.) D3
    evaluation_results_D3 = load_results(
        file_name=result_file_names["FILE_Competitor_D3"], result_folder='Competitor_D3')
    # Calculate mean accuracy of classifier
    accuracy_D3 = np.round(np.mean(evaluation_results_D3["acc_vector"]) * 100, 4)
    # Append accuracy for that experiment to list
    accuracies_experiment["D3"] = accuracy_D3

    # Pay attention about file content (output.txt)!!!
    # 5.) PCA-CD (Qahtan)
    evaluation_results_PCA_CD = load_results(
        file_name=result_file_names["FILE_Competitor_PCA-CD"], result_folder='Competitor_PCA-CD')
    # Calculate mean accuracy of classifier
    accuracy_PCA_CD = np.round(np.mean(evaluation_results_PCA_CD["acc_vector"]) * 100, 4)
    # Append accuracy for that experiment to list
    accuracies_experiment["PCA-CD"] = accuracy_PCA_CD

    # Check if the same amount of data was used so that it is comparable!
    if not (len(evaluation_results_SAW_NewAE["acc_vector"]) == len(evaluation_results_SAW_RetrainAE["acc_vector"])
            == len(evaluation_results_ADWIN10["acc_vector"]) == len(evaluation_results_IBDD["acc_vector"])
            == len(evaluation_results_D3["acc_vector"])):
        print("Error: Files do not have the same number of test data!")
        sys.exit()

    # Append accuracies of experiment to list of all experiments
    accuracies.append(accuracies_experiment)

# Prepare mean row
experiments.append("Mean")
accuracies.append({})
# Create data frame
accuracies_table = pd.DataFrame(data=accuracies, index=[experiments])
# Calculate Mean
accuracies_table.loc['Mean'] = [accuracies_table.mean()]
accuracies_table.to_csv('EvaluationFiles/Competitors/EXPERIMENT_' + str(experiment) + '_EVALUATION_Competitors.csv')

# Generate LaTeX Table
for row in accuracies_table.index.get_level_values(0).unique():
    bolded_value = 'max'
    accuracies_table.loc[row] = [pd.Series(
        data=bold_extreme_values(accuracies_table.loc[[row]], col=False, bolded_value=bolded_value),
    )]
print(accuracies_table.to_latex(escape=False, float_format="%.2f"))



