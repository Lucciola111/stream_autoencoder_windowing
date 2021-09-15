import numpy as np
from Utility_Functions.Plots.PlotReconstructionErrorForPoint import plot_reconstruction_error_for_point

file_where_all = "SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_1DR_100Dims_10MinDimBroken_300MinL_2000MaxL_2021-08-31_13.26.pickle_2021-08-31_16.50_WHERE_ALL_fitNewAETrue_fitFalse_singleGDFalse_initNewADWINTrue_feedNewADWINFalse"
file_where_all_old = "SAW_Autoencoder_ADWIN_Training_RandomNumpyRandomNormalUniform_1DR_100Dims_10MinDimBroken_300MinL_2000MaxL_2021-08-31_13.26.pickle_2021-08-31_16.50_WHERE_OLDPATTERN_fitNewAETrue_fitFalse_singleGDFalse_initNewADWINTrue_feedNewADWINFalse"

results_where_all = np.genfromtxt(file_where_all + '.csv', delimiter=',')
results_where_all_old = np.genfromtxt(file_where_all_old + '.csv', delimiter=',')
extractor_name = "NAE"
detector_name = "IAW"

# Plot ADWIN
# name = 'ADWIN_SAW_Drift_1000_Drift_4000_' + str(extractor_name) + str(detector_name) + '.pdf'
# if initialize_new_adwin_after_drift and feed_adwin_with_refreshed_re_after_drift:
#     title = 'Drift detection with refilling the adaptive window after drift (RAW)'
# elif initialize_new_adwin_after_drift:
#     title = 'Drift detection with initialization of adaptive window after drift (IAW)'
# else:
#     title = 'Drift detection with keeping the adaptive window after drift (KAW)'
# name = False
# plot_adwin(widths, plot_file_name=name, plot_title=title, latex_font=True)

# Plot Reconstruction Error over Time
# name = 'ReconstructionError_SAW_Drift_1000_Drift_4000_' + str(extractor_name) + str(detector_name) + '.pdf'
# title = 'Reconstruction error over time with ' + str(detector_name) + ' after drift'
# name = False
# plot_reconstruction_error_over_time(reconstruction_errors_time,
#                                     plot_file_name=name, plot_title=title, latex_font=True)

# Plot Refreshed Reconstruction Error over Time
# name = 'RefreshedReconstructionError_SAW_Drift_1000_Drift_4000_' + str(extractor_name) + str(detector_name) + '.pdf'
# title = 'Refreshed Reconstruction error over time with ' + str(detector_name) + ' after drift'
# refreshed = True if feed_adwin_with_refreshed_re_after_drift else False
# name = False
# plot_reconstruction_error_over_time(refreshed_reconstruction_errors_time,
#                                     plot_file_name=name, plot_title=title, refreshed=refreshed, latex_font=True)

# Plot reconstruction error of first detected drift
# drift_point_idx = 0
# name = 'ErrorPerDim_FirstDrift_SAW_Drift_1000_Drift_4000_' + str(extractor_name) + str(detector_name) + '.pdf'
# title = 'Reconstruction error per dimension of drift point ' + str(detected_drift_points[drift_point_idx]) + ' with ' + str(extractor_name) + str(detector_name) + ' after drift'
# name = False
#
# plot_reconstruction_error_for_point(error_per_dimension=errors_drift_points_per_dimension[drift_point_idx],
#                                           drift_point=detected_drift_points[drift_point_idx],
#                                           plot_file_name=name, plot_title=title, latex_font=True)

# Plot reconstruction error of specific point
point = 1040
name = 'ErrorPerDim_Point' + str(point) + '_SAW_Drift_1000_Drift_4000_' + str(extractor_name) + str(detector_name) + '.pdf'
name = False
name = 'Figure_5_OldAE_ErrorPerDim_Point1050_SAW_Drift_1000_Drift_4000_NAE-IAW.pdf'
plot_reconstruction_error_for_point(error_per_dimension=results_where_all_old[point],
                                          drift_point=point,
                                          plot_file_name=name, plot_title=False, latex_font=True)

