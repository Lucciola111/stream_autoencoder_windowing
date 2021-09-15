import numpy as np
import re
from Utility_Functions.Plots.PlotAccuracyWithDrifts import plot_accuracy_with_drifts
from Utility_Functions.Plots.PlotADWIN import plot_adwin
from Utility_Functions.Plots.PlotDriftIndicatingDimensionsForPoint import plot_drift_indicating_dimensions_for_point


FILE_all = 'Baseline_MultipleADWINs_StarLightCurves_agreementRate0.1'
# FILE_widths_multiple_adwins = False
FILE_widths_multiple_adwins = 'Baseline_MultipleADWINs_StarLightCurves_widthsMultipleADWINs_agreementRate0.1'
# FILE_drift_points_multiple_adwins = False
FILE_drift_points_multiple_adwins = 'Baseline_MultipleADWINs_StarLightCurves_driftPointsMultipleAdwins_agreementRate0.1'

validation_classifier = True if (np.loadtxt(
    '../Baseline_MultipleADWINs/' + str(FILE_all) + '.csv', delimiter=',').shape[1] == 2) else False
if validation_classifier:
    drift_decisions, acc_vector = np.loadtxt(
        '../Baseline_MultipleADWINs/' + str(FILE_all) + '.csv', delimiter=',', unpack=True, dtype=np.dtype(
            [('drift_decisions', 'bool'), ('acc_vector', 'int')]))
else:
    drift_decisions = np.loadtxt(
        '../Baseline_MultipleADWINs/' + str(FILE_all) + '.csv', delimiter=',', unpack=True, dtype=np.dtype(
            [('drift_decisions', 'bool')]))

if FILE_widths_multiple_adwins:
    widths_multiple_adwins = np.loadtxt(
        '../Baseline_MultipleADWINs/' + str(FILE_widths_multiple_adwins) + '.csv', delimiter=',', dtype='int')

if FILE_drift_points_multiple_adwins:
    drift_points_multiple_adwins = np.loadtxt(
        '../Baseline_MultipleADWINs/' + str(FILE_drift_points_multiple_adwins) + '.csv', delimiter=',', dtype='int')

# Fill array with indices of drift points
drift_points = list(np.where(drift_decisions)[0])

if validation_classifier:
    # Plot Accuracy of evaluation classifier with the drifts
    agreement_rate = float(re.search(r'(?:[_])agreementRate(\d+.\d+)', FILE_all).group(1))
    plot_accuracy_with_drifts(acc_vector, 100, 'ADWIN-' + str(int(agreement_rate*100)), drift_points, None, '-',
                              plot_file_name=(str(FILE_all) + '.pdf')
                              )

# Plot ADWIN for a specified dimension
examined_dimension = 60
plot_adwin(widths_multiple_adwins[examined_dimension],
           plot_file_name=(str(FILE_widths_multiple_adwins) + 'examinedDimension' + str(examined_dimension) + '.pdf')
           )

# Plot which dimensions indicate a drift of first decision for detected drift
if len(drift_points) > 0:
    examined_point = drift_points[0]
    error_per_dimension = list(map(int, drift_points_multiple_adwins[examined_point]))
    plot_drift_indicating_dimensions_for_point(
        point=examined_point, error_per_dimension=error_per_dimension,
        plot_file_name=(str(FILE_drift_points_multiple_adwins) + 'examinedDriftPoint' + str(examined_point) + '.pdf')
        )
