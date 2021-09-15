import pandas as pd
from Evaluation.Plot_DetectionRate_DetectionTime import plot_detectionrate_detectiontime


# 0. Select dataset
normal_distribution = False

# 1. Read in file names of experiment
if normal_distribution:
    experiments = ['onlyMeanDrift_VerySmallVariance_50DR_RandomNormal_Min1DimsBroken',
                   'onlyMeanDrift_SmallVariance_50DR_RandomNormal_Min1DimsBroken',
                   'onlyMeanDrift_HigherVariance_50DR_RandomNormal_Min1DimsBroken',
                   'onlyVarianceDrift_50DR_RandomNormal_Min1DimsBroken',
                   '50DR_RandomNormal_Min100DimsBroken',
                   '50DR_RandomNormal_Min1DimsBroken']
    experiments_names = ['1-MeanDriftVar0.01', '2-MeanDriftVar0.05', '3-MeanDriftVar0.25',
                         '4-VarDrift', '5-MeanVarDriftAll', '6-MeanVarDrift']
    identifier = "NormalDistribution"
else:
    experiments = ['50DR_RandomRandomRBF_Min1DriftCentroid_0to1CS',
                   '100DR_Merged_Min1Broken',
                   '50DR_RandomMNIST_(Fashion)MNIST_SortAllNumbers']
    experiments_names = ["7-RandomRBF", "8-MergedStream", "R1-(F-)MNIST"]
    identifier = "OtherDatasets"

all_experiment_performance_metrics = []
for experiment in experiments:
    experiment_performance_metrics = pd.read_csv(
        'EvaluationFiles/Framework/EXPERIMENT_' + str(experiment) + '_EVALUATION_Framework.csv')
    all_experiment_performance_metrics.append(experiment_performance_metrics)

# 2. Create plot
legend_heading_indices = [0, 7] if normal_distribution else [0, 4]

plot_file_name = 'Figure_5_DetectionRate_DetectionTime_' + str(identifier) + '.pdf'
plot_detectionrate_detectiontime(
    performance_metrics_plot=all_experiment_performance_metrics, experiment_names_plot=experiments_names,
    plot_title=False,
    plot_file_name=plot_file_name,
    latex_font=True, legend_heading_indices=legend_heading_indices)



