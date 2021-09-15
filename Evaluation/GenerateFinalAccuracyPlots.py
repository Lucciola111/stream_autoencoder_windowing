import numpy as np
from Evaluation.Plot_Accuracy import plot_accuracy

path = "../Files_Results/SAW_Autoencoder_ADWIN_Training/ProxyEvaluation/"
file_slcurves = "SAW_Autoencoder_ADWIN_Training_StarLightCurves_2021-09-15_18.19_RESULTS_fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse"
file_heartbeats = "SAW_Autoencoder_ADWIN_Training_Heartbeats_2021-09-15_18.26_RESULTS_fitNewAEFalse_fitTrue_initNewADWINTrue_feedNewADWINFalse"

path_IBDD = "../Files_Results/Competitor_IBDD/ProxyEvaluation/"
file_IBDD_slcurves = "Competitor_IBDD_StarLightCurves_2021-09-15_18.50_RESULTS_windowSize500_epsilon3"
file_IBDD_heartbeats = "Competitor_IBDD_Heartbeats_2021-09-15_20.01_RESULTS_windowSize500_epsilon3"

results_slcurves = np.genfromtxt(path + file_slcurves + '.csv', delimiter=',')
results_heartbeats = np.genfromtxt(path + file_heartbeats + '.csv', delimiter=',')

results_IBDD_slcurves = np.genfromtxt(path_IBDD + file_IBDD_slcurves + '.csv', delimiter=',')
results_IBDD_heartbeats = np.genfromtxt(path_IBDD + file_IBDD_heartbeats + '.csv', delimiter=',')

# Option A - SAW
acc_vector = results_slcurves[:, 3]
detected_drift_points = list(np.where(results_slcurves[:, 2])[0])
method_name = "SAW (NAE-IAW)"
plot_file_name = "Figure_5_ProxyEvaluation_SAW_RAE-IAW_StarLightCurves.pdf"
plot_accuracy(acc_vector, method_name, drifts=detected_drift_points, latex_font=True, plot_file_name=plot_file_name)

acc_vector = results_heartbeats[:, 3]
detected_drift_points = list(np.where(results_heartbeats[:, 2])[0])
method_name = "SAW (NAE-IAW)"
plot_file_name = "Figure_5_ProxyEvaluation_SAW_RAE-IAW_Heartbeats.pdf"
plot_accuracy(acc_vector, method_name, drifts=detected_drift_points, latex_font=True, plot_file_name=plot_file_name)

# Option B - IBDD
acc_vector = results_IBDD_slcurves[:, 1]
detected_drift_points = list(np.where(results_IBDD_slcurves[:, 0])[0])
method_name = "IBDD"
plot_file_name = "Figure_5_ProxyEvaluation_IBDD_StarLightCurves.pdf"
plot_accuracy(acc_vector, method_name, drifts=detected_drift_points, latex_font=True, plot_file_name=plot_file_name)

acc_vector = results_IBDD_heartbeats[:, 1]
detected_drift_points = list(np.where(results_IBDD_heartbeats[:, 0])[0])
method_name = "IBDD"
plot_file_name = "Figure_5_ProxyEvaluation_IBDD_Heartbeats.pdf"
plot_accuracy(acc_vector, method_name, drifts=detected_drift_points, latex_font=True, plot_file_name=plot_file_name)
