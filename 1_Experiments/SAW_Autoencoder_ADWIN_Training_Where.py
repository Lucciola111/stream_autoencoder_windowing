import sys
import tensorflow as tf
import numpy as np
from time import process_time as timer
from sklearn.ensemble import RandomForestClassifier
from skmultiflow.drift_detection.adwin import ADWIN
from TrainingFunctions.LoadDataStream import load_data_stream
from TrainingFunctions.PreprocessingWithLabels import preprocessing_with_labels
from TrainingFunctions.PreprocessingWithoutLabels import preprocessing_without_labels
from TrainingFunctions.BuildAutoencoder import build_autoencoder
from TrainingFunctions.DetectDriftADWINOnlineWithWhere import detect_drift_adwin_online_with_where
from EvaluationFunctions.GeneratePerformanceMetrics import generate_performance_metrics
from Utility_Functions.Plots.PlotADWIN import plot_adwin
from Utility_Functions.Plots.PlotReconstructionErrorForPoint import plot_reconstruction_error_for_point
from Utility_Functions.Plots.PlotReconstructionErrorOverTime import plot_reconstruction_error_over_time
from Utility_Functions.Plots.PlotAutoencoderLoss import plot_autoencoder_loss
from Utility_Functions.Plots.PlotAutoencoderLossOverTime import plot_autoencoder_loss_over_time
from Utility_Functions.Plots.PlotAccuracyWithDrifts import plot_accuracy_with_drifts
from Utility_Functions.CreateResultsFileName import create_results_file_name
from Utility_Functions.GPU import setup_machine

setup_machine(cuda_device=0)

# 0. Set all parameters
# 0.1 Set parameters for autoencoder
percentage_size_encoder = 0.5
epochs = 100
batch_size = 64
early_stopping = True
# 0.2 Set parameters for drift detection
delta_adwin = 0.002
tumbling_window_size = batch_size
drift_epochs = 100
drift_batch_size = 32
# 0.3 Framework decisions
# 0.3.1 Determine training of autoencoder after drift
fit_new_ae_after_drift = False
fit_after_drift = True
# 0.3.2 Update of ADWIN after drift
initialize_new_adwin_after_drift = True
feed_adwin_with_refreshed_re_after_drift = False

if sum([fit_new_ae_after_drift, fit_after_drift]) != 1:
    print("Error: Choose only one option for training after drift!")
    sys.exit()

# 1. Load Data
# Differentiate whether data is provided in separate train and test file
separate_train_test_file = False
image_data = False
drift_labels_known = True
proxy_evaluation = False

if separate_train_test_file:
    path = "IBDD_Datasets/benchmark_real/"

    dataset = "Yoga"
    # dataset = "StarLightCurves"
    # dataset = "Heartbeats"
elif image_data:
    path = "Generated_Streams/Image_Data_Drift_And_Classifier_Labels/"

    dataset = "RandomMNIST_and_FashionMNIST_SortAllNumbers19DR_2021-08-06_11.07.pickle"
else:
    if drift_labels_known:
        if proxy_evaluation:
            path = "Generated_Streams/Drift_And_Classifier_Labels/"

            dataset = "RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids_300MinL_2000MaxL_2021-08-06_10.57.pickle"
        else:
            path = "Generated_Streams/Drift_Labels/"

            # Experiments Evaluation
            dataset = "RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.01_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-10_10.32.pickle"
            # dataset = "RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.05_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.42.pickle"
            # dataset = "RandomNumpyRandomNormalUniform_onlyMeanDrift_var0.25_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.45.pickle"
            # dataset = "RandomNumpyRandomNormalUniform_onlyVarianceDrift_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_11.15.pickle"
            # dataset = "RandomNumpyRandomNormalUniform_50DR_100Dims_100MinDimBroken_300MinL_2000MaxL_2021-08-06_10.54.pickle"
            # dataset = "RandomNumpyRandomNormalUniform_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.53.pickle"
            # dataset = "Mixed_300MinDistance_DATASET_A_RandomNumpyRandomNormalUniform_50DR_100Dims_1MinDimBroken_DATASET_B_RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids.pickle"

            # Experiments Time Complexity
            # "Time_RandomNumpyRandomNormalUniform_10DR_10Dims_1MinDimBroken_300MinL_2000MaxL_2021-09-06_22.24.pickle",
            # "Time_RandomNumpyRandomNormalUniform_10DR_50Dims_5MinDimBroken_300MinL_2000MaxL_2021-09-06_22.25.pickle",
            # "Time_RandomNumpyRandomNormalUniform_10DR_100Dims_10MinDimBroken_300MinL_2000MaxL_2021-09-06_22.26.pickle",
            # "Time_RandomNumpyRandomNormalUniform_10DR_500Dims_50MinDimBroken_300MinL_2000MaxL_2021-09-06_22.26.pickle",
            # "Time_RandomNumpyRandomNormalUniform_10DR_1000Dims_100MinDimBroken_300MinL_2000MaxL_2021-09-06_22.27.pickle"

    else:
        path = "Generated_Streams/Classifier_Labels/"
        dataset = ""

# Load data stream
data_stream = load_data_stream(dataset=dataset, path=path, separate_train_test_file=separate_train_test_file,
                               image_data=image_data, drift_labels_known=drift_labels_known,
                               proxy_evaluation=proxy_evaluation)
# Set number of instances
n_instances = data_stream.shape[0]
# Set number of train data, validation data, and test data
n_train_data = 1000
n_val_data = 0
n_test_data = int(n_instances - n_val_data - n_train_data)

# 2. Pre-processing
# Separate data stream and drift labels
if drift_labels_known:
    data_stream, drift_labels = data_stream[:, :-1], data_stream[:, -1]
    drift_labels = drift_labels[(len(drift_labels) - n_test_data):]
# Preprocess data stream
if proxy_evaluation:
    train_X, train_y, val_X, val_y, test_X, test_y = preprocessing_with_labels(
        data_stream, n_instances, n_train_data, n_val_data, n_test_data, image_data)
else:
    train_X, val_X, test_X = preprocessing_without_labels(
        data_stream, n_instances, n_train_data, n_val_data, n_test_data)


# Set number of dimensions
n_dimensions = train_X.shape[1]
# Set size of encoder
size_encoder = int(percentage_size_encoder * n_dimensions)

# Convert to tensors
train_X = tf.convert_to_tensor(train_X, np.float32)
test_X = tf.convert_to_tensor(test_X, np.float32)

# 3. Train classifier for Evaluation
if proxy_evaluation:
    model_classifier = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
    # train_X = np.concatenate((train_X, val_X), axis=0)
    # train_y = np.concatenate((train_y, val_y), axis=0)
    model_classifier.fit(train_X, train_y)
    acc_vector = np.zeros(len(test_y), dtype=int)

# 4. Dimension Reduction: Autoencoder
# 4.1 Check whether bottleneck of autoencoder is smaller than dimensions of dataset
if size_encoder >= n_dimensions:
    print("Error: Bottleneck of autoencoder larger than number of dimensions!")
    sys.exit()

start = timer()
# 4.2 Build autoencoder
autoencoder, encoder, decoder = build_autoencoder(n_dimensions=n_dimensions, size_encoder=size_encoder)
# Callback will stop training when there is no improvement in validation loss for three consecutive epochs.
callback = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)] if early_stopping else None
# 4.3 Train autoencoder
autoencoder.fit(train_X, train_X,
                epochs=epochs, batch_size=batch_size,
                shuffle=True, verbose=0,
                callbacks=callback
                )
all_losses = []
all_losses.extend(autoencoder.history.history['loss'])
final_losses = [autoencoder.history.history['loss'][-1]]

# 5. Drift Detection
# Test: Apply ADWIN on Reconstruction Error, Train: Update Autoencoder with new instance(/batch)
# 5.1 Initialize arrays
reconstruction_errors_time = []
refreshed_reconstruction_errors_time = []
autoencoder_loss_time = []
autoencoder_update_loss = False
widths = []
drift_decisions = []
errors_drift_points_per_dimension = []
all_errors_per_dimensions = []
all_errors_per_dimensions_old = []
autoencoder_old, encoder_old, decoder_old = False, False, False
decoded_old = False

# 5.2 Initialize ADWIN
adwin = ADWIN(delta=delta_adwin)
# 5.3 Examine for a tumbling window:
for idx in range(0, n_test_data, tumbling_window_size):
    # 5.3.1 Calculate RE for all elements in Tumbling Window
    new_element = test_X[idx: (idx + tumbling_window_size)]
    # Apply autoencoder to new streaming element/batch
    encoded = encoder.predict(new_element)
    decoded = decoder.predict(encoded)
    # Calculate reconstruction error(s) of new streaming element/batch
    dist = np.linalg.norm(new_element - decoded, axis=1)
    reconstruction_errors_time.extend(dist)
    refreshed_reconstruction_errors_time.extend(dist)
    # Apply autoencoder of old pattern to new streaming batch
    if autoencoder_old:
        encoded_old = encoder_old.predict(new_element)
        decoded_old = decoder_old.predict(encoded_old)
    # Calculate Loss of autoencoder
    loss = autoencoder_update_loss if autoencoder_update_loss else autoencoder.history.history['loss'][-1]
    autoencoder_loss_time.append(loss)

    # 5.3.2: Test - Make prediction for elements with classifier
    if proxy_evaluation:
        y_pred = model_classifier.predict(new_element)
        for element in range(len(new_element)):
            if y_pred[element] == test_y[idx + element]:
                acc_vector[idx + element] = 1

    # If we have validation classifier: Set parameters for retraining classifier after drift
    model_classifier_parameter = model_classifier if proxy_evaluation else False
    test_y_parameter = test_y if proxy_evaluation else False

    # 5.3.3 Detect Drift with ADWIN on reconstruction error
    adwin, batch_widths, batch_drift_decisions, batch_errors_drift_points_per_dimension, \
    batch_all_errors_per_dimension, batch_all_errors_per_dimension_old, fusioned_refreshed_dist_adwin, \
    all_losses, final_losses, autoencoder, encoder, decoder, \
    autoencoder_old, encoder_old, decoder_old = detect_drift_adwin_online_with_where(
        test_X=test_X, idx=idx, dist=dist, adwin=adwin, autoencoder=autoencoder, encoder=encoder, decoder=decoder,
        decoded=decoded, autoencoder_old=autoencoder_old, encoder_old=encoder_old, decoder_old=decoder_old,
        decoded_old=decoded_old, drift_epochs=drift_epochs, drift_batch_size=drift_batch_size,
        early_stopping=early_stopping, delta_adwin=delta_adwin, all_losses=all_losses, final_losses=final_losses,
        model=model_classifier_parameter, test_y=test_y_parameter, fit_new_ae_after_drift=fit_new_ae_after_drift,
        fit_after_drift=fit_after_drift, initialize_new_adwin_after_drift=initialize_new_adwin_after_drift,
        feed_adwin_with_refreshed_re_after_drift=feed_adwin_with_refreshed_re_after_drift)

    # 5.3.4 If no drift is detected, the autoencoder is updated
    if sum(batch_drift_decisions) == 0:
        # Run single gradient update on current batch
        loss_update = autoencoder.train_on_batch(x=new_element, y=new_element)
        all_losses.append(loss_update)
        final_losses.append(loss_update)
    # 5.3.5 If a new drift is detected, a copy of the autoencoder is set/ the copy is updated
    else:
        # Replace distances with new reconstruction errors
        length_old_list = len(refreshed_reconstruction_errors_time)
        length_new_elements = len(fusioned_refreshed_dist_adwin)
        refreshed_reconstruction_errors_time[(length_old_list - length_new_elements):] = fusioned_refreshed_dist_adwin

    # 5.3.6 Save results from drift detection
    # Save width of ADWIN, and whether a drift occurred,
    # Save for the drift points: reconstruction errors for each dimension
    widths.extend(batch_widths)
    drift_decisions.extend(batch_drift_decisions)
    errors_drift_points_per_dimension.extend(batch_errors_drift_points_per_dimension)
    all_errors_per_dimensions.extend(batch_all_errors_per_dimension)
    all_errors_per_dimensions_old.extend(batch_all_errors_per_dimension_old)

    if idx % 500 == 0:
        print(f"Iteration of Loop: {idx}/{len(test_X)}")

# 6. Evaluation
# Fill array with indices of drift points
detected_drift_points = list(np.where(drift_decisions)[0])
# 6.1 Drift Detection Evaluation
if drift_labels_known:
    # Define acceptance levels
    acceptance_levels = [60, 120, 180, 300]
    # Generate performance metrics
    simple_metrics = False
    complex_metrics = True
    performance_metrics = generate_performance_metrics(
        drift_decisions=drift_decisions, drift_labels=drift_labels, acceptance_levels=acceptance_levels,
        simple_metrics=simple_metrics, complex_metrics=complex_metrics, index_name='SAW')

# 6.2 Proxy Evaluation
if proxy_evaluation:
    # Calculate mean accuracy of classifier
    mean_acc = np.mean(acc_vector) * 100
    print('Average classification accuracy: {}%'.format(np.round(mean_acc, 2)))
# Measure the elapsed time
end = timer()
execution_time = end - start
print('Time per example: {} sec'.format(np.round(execution_time / len(test_X), 4)))
print('Total time: {} sec'.format(np.round(execution_time, 2)))


# 7. Plot Results
# Prepare abbreviations for plot title and plot file name

# Plot Accuracy of evaluation classifier with the drifts
if proxy_evaluation:
    plot_accuracy_with_drifts(acc_vector, 100, "SAW", detected_drift_points, None, '-')

# Plot ADWIN
title = "Adaptive window size"
plot_adwin(widths, plot_file_name=False, plot_title=title, latex_font=True)

# Plot Reconstruction Error over Time
title = 'Reconstruction error over time'
plot_reconstruction_error_over_time(reconstruction_errors_time,
                                    plot_file_name=False, plot_title=title, latex_font=True)

# Plot Refreshed Reconstruction Error over Time
title = 'Refreshed Reconstruction error over time'
refreshed = True if feed_adwin_with_refreshed_re_after_drift else False
plot_reconstruction_error_over_time(refreshed_reconstruction_errors_time,
                                    plot_file_name=False, plot_title=title, refreshed=refreshed, latex_font=True)

# Plot reconstruction error of first detected drift
if len(detected_drift_points) > 0:
    drift_point_idx = 0
    title = 'Reconstruction error per dimension of drift point ' + str(detected_drift_points[drift_point_idx])
    plot_reconstruction_error_for_point(error_per_dimension=errors_drift_points_per_dimension[drift_point_idx],
                                        drift_point=detected_drift_points[drift_point_idx],
                                        plot_file_name=False, plot_title=title, latex_font=True)

# Plot reconstruction error of specific point
if len(all_errors_per_dimensions) > 0:
    point = 1000
    title = 'Reconstruction error per dimension of point ' + str(point)
    plot_reconstruction_error_for_point(error_per_dimension=all_errors_per_dimensions[point],
                                        drift_point=point, plot_file_name=False, plot_title=title, latex_font=True)

# Plot Loss of autoencoder
plot_autoencoder_loss(autoencoder=autoencoder, val_data=False)
# Plot Loss of autoencoder over time
# plot_autoencoder_loss_over_time(autoencoder_loss_time=autoencoder_loss_time)

# Plot loss of all trainings over time
title = 'Autoencoder loss after each gradient decent step with'
plot_autoencoder_loss_over_time(autoencoder_loss_time=all_losses,
                                plot_file_name=False, plot_title=title)
# plot_autoencoder_loss_over_time(autoencoder_loss_time=final_losses, plot_title="Autoencoder Loss after each training")


# 8. Save results in file
# Create file name
file_name = create_results_file_name(
    dataset=dataset, algorithm_name='SAW_Autoencoder_ADWIN_Training', drift_labels_known=drift_labels_known,
    proxy_evaluation=proxy_evaluation, image_data=image_data)

# Save values for: widths, reconstruction_errors_time, drift_decisions, acc_vector
if proxy_evaluation and drift_labels_known:
    result_data = np.transpose([widths, reconstruction_errors_time, drift_decisions, acc_vector, drift_labels])
    result_header = "widthsADWIN, reconstructionErrorTime, driftDecisions, accVector, driftLabels"
    result_format = ('%i', '%1.12f', '%i', '%i', '%i')
elif proxy_evaluation:
    result_data = np.transpose([widths, reconstruction_errors_time, drift_decisions, acc_vector])
    result_header = "widthsADWIN, reconstructionErrorTime, driftDecisions, accVector"
    result_format = ('%i', '%1.12f', '%i', '%i')
elif drift_labels_known:
    result_data = np.transpose([widths, reconstruction_errors_time, drift_decisions, drift_labels])
    result_header = "widthsADWIN, reconstructionErrorTime, driftDecisions, driftLabels"
    result_format = ('%i', '%1.12f', '%i', '%i')
else:
    result_data = np.transpose([widths, reconstruction_errors_time, drift_decisions])
    result_header = "widthsADWIN, reconstructionErrorTime, driftDecisions"
    result_format = ('%i', '%1.12f', '%i')

# Save values for: widths, reconstruction_errors_time, drift_decisions, acc_vector
np.savetxt(str(file_name) + '_RESULTS'
           + '_fitNewAE' + str(fit_new_ae_after_drift)
           + '_fit' + str(fit_after_drift)
           + '_initNewADWIN' + str(initialize_new_adwin_after_drift)
           + '_feedNewADWIN' + str(feed_adwin_with_refreshed_re_after_drift)
           + '.csv', result_data, header=result_header, delimiter=",", fmt=result_format)

# Save reconstruction errors of drift point per dimension
np.savetxt(str(file_name) + '_WHERE'
           + '_fitNewAE' + str(fit_new_ae_after_drift)
           + '_fit' + str(fit_after_drift)
           + '_initNewADWIN' + str(initialize_new_adwin_after_drift)
           + '_feedNewADWIN' + str(feed_adwin_with_refreshed_re_after_drift)
           + '.csv', errors_drift_points_per_dimension, delimiter=",")

# For "WHERE" Evaluation
# Save reconstruction errors of drift point per dimension
np.savetxt(str(file_name) + '_WHERE_ALL'
           + '_fitNewAE' + str(fit_new_ae_after_drift)
           + '_fit' + str(fit_after_drift)
           + '_initNewADWIN' + str(initialize_new_adwin_after_drift)
           + '_feedNewADWIN' + str(feed_adwin_with_refreshed_re_after_drift)
           + '.csv', batch_all_errors_per_dimension, delimiter=",")

# For "WHERE" Evaluation
# Save reconstruction errors of drift point per dimension
np.savetxt(str(file_name) + '_WHERE_OLDPATTERN'
           + '_fitNewAE' + str(fit_new_ae_after_drift)
           + '_fit' + str(fit_after_drift)
           + '_initNewADWIN' + str(initialize_new_adwin_after_drift)
           + '_feedNewADWIN' + str(feed_adwin_with_refreshed_re_after_drift)
           + '.csv', batch_all_errors_per_dimension_old, delimiter=",")

# Add parameter information to txt file
file = open(file_name + '.txt', "w")
file.write('PARAMETERS: ' + 'sizeEncoder' + str(size_encoder)
           + '_initialEpochs' + str(epochs)
           + '_initialBatchSize' + str(batch_size)
           + '_afterDriftEpochs' + str(drift_epochs)
           + '_afterDriftBatchSize' + str(drift_batch_size)
           + '_earlyStopping' + str(early_stopping)
           + '_deltaADWIN' + str(delta_adwin)
           + '_tumblingWindowSize' + str(tumbling_window_size)
           + '_numTrainData' + str(n_train_data)
           + '_numValData' + str(n_val_data)
           + '_num_TestData' + str(n_test_data) + '\n')
file.write('FRAMEWORK: ' + 'sizeEncoder' + str(size_encoder)
           + '_fitNewAE' + str(fit_new_ae_after_drift)
           + '_fit' + str(fit_after_drift)
           + '_initNewADWIN' + str(initialize_new_adwin_after_drift)
           + '_feedNewADWIN' + str(feed_adwin_with_refreshed_re_after_drift) + '\n')
file.close()


detected_drift_points = list(np.where(drift_decisions)[0])
if drift_labels_known:
    actual_drift_points = list(np.where(drift_labels)[0])
