import numpy as np
from Utility_Functions.Plots.PlotAccuracyWithDrifts import plot_accuracy_with_drifts
from Utility_Functions.Plots.PlotReconstructionErrorOverTime import plot_reconstruction_error_over_time

FILE_all = "Baseline_DynamicAutoencoder_StaticClassifier_StarLightCurves_withValidationTrue_sizeEncoder60_epochs40_batchSize64"
# FILE_all = "Baseline_DynamicAutoencoder_StaticClassifier_data_stream_random_rbf_changespeed2.87_numdriftcentroids30.csv_withValidationTrue_sizeEncoder60_epochs40_batchSize64"

validation_classifier = True if (np.loadtxt(
    '../Baseline_StaticClassifier/' + str(FILE_all) + '.csv', delimiter=',').shape[1] == 2) else False

if validation_classifier:
    reconstruction_error_time, acc_vector = np.loadtxt(
        '../Baseline_StaticClassifier/' + str(FILE_all) + '.csv', delimiter=',', unpack=True,
        dtype=np.dtype([('reconstruction_error_time', 'float'), ('acc_vector', 'int')]))
else:
    reconstruction_error_time = np.loadtxt(
        '../Baseline_StaticClassifier/' + str(FILE_all) + '.csv', delimiter=',', unpack=True,
        dtype=np.dtype([('reconstruction_error_time', 'float')]))

if validation_classifier:
    # Plot Accuracy of evaluation classifier (without drifts as topline does not detect drifts)
    plot_accuracy_with_drifts(acc_vector, 100, 'Baseline', drifts=False,
                              plot_file_name=(str(FILE_all) + '.pdf')
                              )
    # Calculate mean accuracy of classifier
    mean_acc = np.mean(acc_vector) * 100
    print('Average classification accuracy: {}%'.format(np.round(mean_acc, 2)))
# Plot Reconstruction Error over Time
plot_reconstruction_error_over_time(reconstruction_error_time,
                                    plot_file_name=(str(FILE_all) + '.pdf')
                                    )
