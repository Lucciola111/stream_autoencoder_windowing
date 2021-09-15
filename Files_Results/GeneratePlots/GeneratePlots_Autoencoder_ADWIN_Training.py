import numpy as np
from Utility_Functions.Plots.PlotAccuracyWithDrifts import plot_accuracy_with_drifts
from Utility_Functions.Plots.PlotADWIN import plot_adwin
from Utility_Functions.Plots.PlotReconstructionErrorOverTime import plot_reconstruction_error_over_time
from Utility_Functions.Plots.PlotReconstructionErrorForDriftPoint import plot_reconstruction_error_for_drift_point

# FILE_all = 'Autoencoder_ADWIN_Training_StarLightCurves_withValidationTrue_sizeEncoder60_epochs40_batchSize64_tumblingWindowSize64_newFitAfterDriftTrue'
# FILE_all = 'Autoencoder_ADWIN_Training_StarLightCurves_withValidationTrue_sizeEncoder60_epochs40_batchSize64_tumblingWindowSize1_newFitAfterDriftTrue'
FILE_all = 'Autoencoder_ADWIN_Training_data_stream_numpy_instance3000_dim60.csv_withValidationFalse_sizeEncoder60_epochs40_batchSize64_tumblingWindowSize64_newFitAfterDriftTrue'

# FILE_errors_drift_points_per_dimension = False
# FILE_errors_drift_points_per_dimension = 'Autoencoder_ADWIN_Training_StarLightCurves_errorsDriftPointsPerDimension_sizeEncoder60_epochs40_batchSize64_tumblingWindowSize64_newFitAfterDriftTrue'
# FILE_errors_drift_points_per_dimension = 'Autoencoder_ADWIN_Training_StarLightCurves_errorsDriftPointsPerDimension_sizeEncoder60_epochs40_batchSize64_tumblingWindowSize1_newFitAfterDriftTrue'
FILE_errors_drift_points_per_dimension = 'Autoencoder_ADWIN_Training_data_stream_numpy_instance3000_dim60.csv_errorsDriftPointsPerDimension_sizeEncoder60_epochs40_batchSize64_tumblingWindowSize64_newFitAfterDriftTrue'


validation_classifier = True if (np.loadtxt(
    '../SAW_Autoencoder_ADWIN_Training/' + str(FILE_all) + '.csv', delimiter=',').shape[1] == 4) else False

if validation_classifier:
    widths, reconstruction_error_time, drift_decisions, acc_vector = np.loadtxt(
        '../SAW_Autoencoder_ADWIN_Training/' + str(FILE_all) + '.csv', delimiter=',', unpack=True, dtype=np.dtype(
            [('widths', 'int'), ('reconstruction_error_time', 'float'), ('drift_decisions', 'bool'), ('acc_vector', 'int')]
        ))
else:
    widths, reconstruction_error_time, drift_decisions = np.loadtxt(
        '../SAW_Autoencoder_ADWIN_Training/' + str(FILE_all) + '.csv', delimiter=',', unpack=True,
        dtype=np.dtype([('widths', 'int'), ('reconstruction_error_time', 'float'), ('drift_decisions', 'bool')]))

# Fill array with indices of drift points
drift_points = list(np.where(drift_decisions)[0])

# Plot ADWIN
plot_adwin(adwin=widths,
           plot_file_name=(str(FILE_all) + '.pdf'),
           latex_font=False
           )
# Plot Reconstruction Error over Time
plot_reconstruction_error_over_time(reconstruction_error_time,
                                    plot_file_name=(str(FILE_all) + '.pdf'),
                                    latex_font=False
                                    )

if validation_classifier:
    # Plot Accuracy of evaluation classifier with the drifts
    plot_accuracy_with_drifts(acc_vector, 100, 'ADWIN-HDD', drift_points,
                              plot_file_name=(str(FILE_all) + '.pdf'),
                              latex_font=False
                              )
    # Calculate mean accuracy of classifier
    mean_acc = np.mean(acc_vector) * 100
    print('Average classification accuracy: {}%'.format(np.round(mean_acc, 2)))

if FILE_errors_drift_points_per_dimension and len(drift_points) > 0:
    errors_drift_points_per_dimension = np.loadtxt(
        '../SAW_Autoencoder_ADWIN_Training/' + str(FILE_errors_drift_points_per_dimension) + '.csv', delimiter=',')
    # Plot reconstruction error of first detected drift
    drift_point_idx = 0

    plot_reconstruction_error_for_drift_point(
        error_per_dimension=errors_drift_points_per_dimension[drift_point_idx],
        drift_point=drift_points[drift_point_idx],
        plot_file_name=(str(FILE_errors_drift_points_per_dimension) + '_driftPointIdx' + str(drift_point_idx) + '.pdf'),
        latex_font=False
        )


