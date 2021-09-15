import numpy as np
import pandas as pd
from EvaluationFunctions.LoadFrameworkDesignsFilenames import load_framework_designs_filenames
from EvaluationFunctions.LoadCompetitorsFilenames import load_competitors_filenames
from EvaluationFunctions.Load_withIterations_Results import load_with_iterations_results
from Evaluation.Plot_TimeComplexity import plot_time_complexity

# 0. Read in file names of experiment
experiments = ["10Dim", "50Dim", "100Dim", "500Dim", "1000Dim"]
competitors = True

times_per_example = []
df_times_per_example = pd.DataFrame()
for experiment in experiments:
    time_per_example_experiment = {}
    # 1. Read in File names
    if competitors:
        path, dataset, result_file_names = load_competitors_filenames(experiment=experiment)

        file_names = ["FILE_SAW_NewAE", "FILE_SAW_RetrainAE", "FILE_Baseline_ADWIN10", "FILE_Baseline_ADWIN10-initialized",
                      "FILE_Competitor_IBDD", "FILE_Competitor_D3"]
        result_folders = ["SAW_Autoencoder_ADWIN_Training", "SAW_Autoencoder_ADWIN_Training", "Baseline_MultipleADWINS",
                          "Baseline_MultipleADWINS", "Competitor_IBDD", "Competitor_D3"]
        experiment_names = ["SAW (NAE-IAW)", "SAW (RAE-IAW)", "ADWIN-10", "ADWIN-10i", "IBDD", "D3"]
    else:
        path, dataset, result_file_names = load_framework_designs_filenames(experiment=experiment)

        file_names = ["FILE_TrainNewAE_KeepADWIN", "FILE_TrainNewAE_InitializeADWIN",
                      "FILE_TrainNewAE_InitializeAndFeedADWIN",
                      "FILE_RetrainAE_KeepADWIN", "FILE_RetrainAE_InitializeADWIN", "FILE_RetrainAE_InitializeAndFeedADWIN"]
        result_folders = ["SAW_Autoencoder_ADWIN_Training"] * 6
        experiment_names = ["NAE-KAW", "NAE-IAW", "NAE-RAW", "RAE-KAW", "RAE-IAW", "RAE-RAW"]

    # 2. Read in Files and generate evaluation metrics
    for experiment_idx in range(len(file_names)):
        if result_file_names[file_names[experiment_idx]] != '-':
            evaluation_results = load_with_iterations_results(
                file_name=result_file_names[file_names[experiment_idx]], result_folder=result_folders[experiment_idx])

            time_per_example = np.round(np.mean(evaluation_results['Time per Example']), 4)
            time_per_example_experiment[experiment_names[experiment_idx]] = time_per_example
        else:
            time_per_example_experiment[experiment_names[experiment_idx]] = 0

    # Append accuracies of experiment to list of all experiments
    times_per_example.append(time_per_example_experiment)

# Create data frame
times_per_example_table = pd.DataFrame(data=times_per_example, index=experiments)

times_per_example_table.to_csv('EvaluationFiles/Competitors/EXPERIMENT_'
                               + str(experiment) + '_TIMECOMPLEXITY_EVALUATION_Competitors.csv')

if competitors:
    plot_file_name = "Figure_5_TimeComplexity_Competitors"
else:
    plot_file_name = "Figure_5_TimeComplexity_FrameworkDesign"

plot_time_complexity(data=times_per_example_table, competitors=competitors, plot_file_name=plot_file_name, latex_font=True)

