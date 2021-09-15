import re
import pandas as pd
from SensitivityStudy.LoadSensitivityStudyFileNames import load_sensitivity_study_file_names
from Utility_Functions.BoldExtremeValues import bold_extreme_values


# 0.1 Selection design
design = "NAE-IAW"
# design = "RAE-IAW"

# 0.2 Read in file names of design
path, design_file_names = load_sensitivity_study_file_names(design=design)

file_names = ['FILE_MeanDrift_var0.01', 'FILE_MeanDrift_var0.05', 'FILE_MeanDrift_var0.25',
              'FILE_VarianceDrift', 'FILE_MeanVarianceDrift_all_broken', 'FILE_MeanVarianceDrift',
              'FILE_RandomRBF_Generator', 'FILE_MergedStream', 'FILE_FashionMNIST'
              ]
experiment_names = ["1-MeanDriftVar0.01", "2-MeanDriftVar0.05", "3-MeanDriftVar0.25",
                    "4-VarDrift", "5-MeanVarDriftAll", "6-MeanVarDrift",
                    "7-RandomRBF", "8-Merged Stream", "R1-(F-)MNIST"
                    ]

# 1. Iterate through experiments to load evaluation results
all_experiments = []
for experiment_idx in range(len(file_names)):
    # Load evaluation results
    file_name = file_names[experiment_idx]
    experiment_encoder_values_results = pd.read_csv(path + str(design_file_names[file_name]) + '.csv', index_col=0)
    experiment_encoder_values_results.rename(index={0: experiment_names[experiment_idx]}, inplace=True)

    all_experiments.append(experiment_encoder_values_results)

# 2. Prepare mean and median row
all_experiments.append(pd.DataFrame(index=['Mean']))
all_experiments.append(pd.DataFrame(index=['Median']))

# Generate data for table of sensitivity study
# Merge performance metrics of experiments
experiments_merged = pd.concat(all_experiments, axis=0)
# Calculate mean
experiments_merged.loc['Mean'] = experiments_merged.mean()
experiments_merged.loc['Median'] = experiments_merged.median()
# Write to csv file
experiments_merged.to_csv('../Evaluation/EvaluationFiles/Sensitivity_Study/DESIGN_' + str(design) + '_SensitivityStudy.csv')

# Generate LaTeX Table
for row in experiments_merged.index.get_level_values(0).unique():
    bolded_value = 'max'
    experiments_merged.loc[row] = bold_extreme_values(experiments_merged.loc[[row]], bolded_value=bolded_value, col=False)
print(re.sub(' +', ' ', experiments_merged.to_latex(escape=False)))
