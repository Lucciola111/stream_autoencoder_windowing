import pandas as pd
from Evaluation.Plot_DetectionRate_DetectionTime import plot_detectionrate_detectiontime


experiments_plot = ["50DR_RandomNormal_CompleteSet_Min10DimsBroken",
                    "50DR_RandomNormal_CompleteSet_AllDimsBroken",
                    "50DR_RandomRandomRBF_CompleteSet_Min1DriftCentroid_0to1CS"]

performance_metrics_plot = []
for experiment in experiments_plot:
    performance_metrics = pd.read_csv("EvaluationFiles/EXPERIMENT_" + str(experiment) + "_EVALUATION.csv", index_col=0)
    performance_metrics_plot.append(performance_metrics)

plot_detectionrate_detectiontime(performance_metrics_plot=performance_metrics_plot,
                                 experiment_names_plot=experiments_plot,
                                 plot_title=False, plot_file_name=False, latex_font=False)
