import numpy as np
from Utility_Functions.Plots.PlotAccuracyWithDrifts import plot_accuracy_with_drifts
from Utility_Functions.Plots.PlotReconstructionErrorOverTime import plot_reconstruction_error_over_time

# CODE FOR OLD FILES:
# FILE_reconstruction_error_time = "Topline_startLightCurves_reconstructionErrorTime_sizeEncoder60_epochs40_batchSize64.csv"
# FILE_acc_vector = "Topline_startLightCurves_accVector_sizeEncoder60_epochs40_batchSize64.csv"
# reconstruction_error_time = np.loadtxt('../Topline/' + str(FILE_reconstruction_error_time), delimiter=',')
# acc_vector = np.loadtxt('../Topline/' + str(FILE_acc_vector), delimiter=',')
# plot_accuracy_with_drifts(acc_vector, 100, 'Topline', plot_file_name=(str(FILE_acc_vector) + '.pdf'))
# plot_reconstruction_error_over_time(reconstruction_error_time, plot_file_name=(str(FILE_reconstruction_error_time) + '.pdf'))

FILE_all = "Topline_StartLightCurves_withValidationTrue_sizeEncoder60_epochs40_batchSize64"
# FILE_all = "Topline_Yoga_withValidationTrue_sizeEncoder60_epochs40_batchSize64"

validation_classifier = True if (np.loadtxt(
    '../Topline/' + str(FILE_all) + '.csv', delimiter=',').shape[1] == 2) else False

if validation_classifier:
    reconstruction_error_time, acc_vector = np.loadtxt(
        '../Topline/' + str(FILE_all) + '.csv', delimiter=',', unpack=True,
        dtype=np.dtype([('reconstruction_error_time', 'float'), ('acc_vector', 'int')]))
else:
    reconstruction_error_time = np.loadtxt(
        '../Topline/' + str(FILE_all) + '.csv', delimiter=',', unpack=True,
        dtype=np.dtype([('reconstruction_error_time', 'float')]))

if validation_classifier:
    # Plot Accuracy of evaluation classifier (without drifts as topline does not detect drifts)
    plot_accuracy_with_drifts(acc_vector, 100, 'Topline', drifts=False,
                              plot_file_name=(str(FILE_all) + '.pdf')
                              )
    # Calculate mean accuracy of classifier
    mean_acc = np.mean(acc_vector) * 100
    print('Average classification accuracy: {}%'.format(np.round(mean_acc, 2)))
# Plot Reconstruction Error over Time
plot_reconstruction_error_over_time(reconstruction_error_time,
                                    plot_file_name=(str(FILE_all) + '.pdf')
                                    )
