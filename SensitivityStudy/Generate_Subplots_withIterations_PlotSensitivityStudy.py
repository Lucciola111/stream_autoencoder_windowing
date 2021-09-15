import pandas as pd
import matplotlib.pyplot as plt
from SensitivityStudy.Plot_Sensitivity_Study import plot_sensitivity_study


result_data_NAE_IAW = pd.read_csv('../Evaluation/EvaluationFiles/Sensitivity_Study/DESIGN_NAE-IAW_SensitivityStudy.csv',
                                  index_col=0)
result_data_RAE_IAW = pd.read_csv('../Evaluation/EvaluationFiles/Sensitivity_Study/DESIGN_RAE-IAW_SensitivityStudy.csv',
                                  index_col=0)

plot_columns = ['Normal distribution', 'Complex datasets', 'Median']
fig, [ax1, ax2, ax3] = plt.subplots(1, 3, sharey=True, figsize=(20, 5))
fig.suptitle('Sensitivity Study')


# Set values and data
for subplot_columns in plot_columns:
    if subplot_columns == 'Median':
        plot_data_NAE_IAW = result_data_NAE_IAW[-1:].T
        plot_data_RAE_IAW = result_data_RAE_IAW[-1:].T
        plot_sensitivity_study(data_NAE_IAW=plot_data_NAE_IAW, data_RAE_IAW=plot_data_RAE_IAW, ax=ax1,
                               plot_file_name=False, plot_title=subplot_columns, latex_font=True)
    if subplot_columns == 'Normal distribution':
        plot_data_NAE_IAW = result_data_NAE_IAW[0:6].T
        plot_data_RAE_IAW = result_data_RAE_IAW[0:6].T
        plot_sensitivity_study(data_NAE_IAW=plot_data_NAE_IAW, data_RAE_IAW=plot_data_RAE_IAW, ax=ax2,
                               plot_file_name=False, plot_title=subplot_columns, latex_font=True)
    if subplot_columns == 'Complex datasets':
        plot_data_NAE_IAW = result_data_NAE_IAW[6:9].T
        plot_data_RAE_IAW = result_data_RAE_IAW[6:9].T
        plot_sensitivity_study(data_NAE_IAW=plot_data_NAE_IAW, data_RAE_IAW=plot_data_RAE_IAW, ax=ax3,
                               plot_file_name=False, plot_title=subplot_columns, latex_font=True)

# Save overall plot
plot_file_name = "Plot_F1Score_Sensitivity_Study_SharedAxis.pdf"
plt.savefig("Plots/" + str(plot_file_name), bbox_inches='tight')
plt.show()
