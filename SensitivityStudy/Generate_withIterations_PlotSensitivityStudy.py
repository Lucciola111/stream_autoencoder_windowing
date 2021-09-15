import pandas as pd
import matplotlib.pyplot as plt
from SensitivityStudy.Plot_Sensitivity_Study import plot_sensitivity_study


# Choose datasets
normal_distribution_part1 = True
normal_distribution_part2 = True
other_datasets = False

result_data_NAE_IAW = pd.read_csv('../Evaluation/EvaluationFiles/Sensitivity_Study/DESIGN_NAE-IAW_SensitivityStudy.csv',
                                  index_col=0)
result_data_RAE_IAW = pd.read_csv('../Evaluation/EvaluationFiles/Sensitivity_Study/DESIGN_RAE-IAW_SensitivityStudy.csv',
                                  index_col=0)

# Set values and data
if normal_distribution_part1 and normal_distribution_part2:
    identifier = "NormalDistribution"
    plot_title = "Normal distribution"
    plot_data_NAE_IAW = result_data_NAE_IAW[0:6].T
    plot_data_RAE_IAW = result_data_RAE_IAW[0:6].T
elif normal_distribution_part1:
    identifier = "MeanDrift"
    plot_title = "Mean drift"
    plot_data_NAE_IAW = result_data_NAE_IAW[0:3].T
    plot_data_RAE_IAW = result_data_RAE_IAW[0:3].T
elif normal_distribution_part2:
    identifier = "(Mean)VarDrift"
    plot_title = "Mean and/or variance drift"
    plot_data_NAE_IAW = result_data_NAE_IAW[3:6].T
    plot_data_RAE_IAW = result_data_RAE_IAW[3:6].T
elif other_datasets:
    identifier = "OtherDatasets"
    plot_title = "Complex datasets"
    plot_data_NAE_IAW = result_data_NAE_IAW[6:9].T
    plot_data_RAE_IAW = result_data_RAE_IAW[6:9].T
else:
    identifier = "Median"
    plot_title = "Median"
    plot_data_NAE_IAW = result_data_NAE_IAW[-1:].T
    plot_data_RAE_IAW = result_data_RAE_IAW[-1:].T

plot_file_name = "Figure_5_Plot_F1Score_Sensitivity_Study_" + str(identifier) + ".pdf"
plot_sensitivity_study(data_NAE_IAW=plot_data_NAE_IAW, data_RAE_IAW=plot_data_RAE_IAW,
                       plot_file_name=plot_file_name, plot_title=plot_title, latex_font=True)

plt.show()
