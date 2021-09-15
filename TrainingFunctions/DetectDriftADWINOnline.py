import tensorflow as tf
import numpy as np
from skmultiflow.drift_detection.adwin import ADWIN
from Utility_Functions.ComputeErrorPerDim import compute_error_per_dim
from TrainingFunctions.BuildAutoencoder import build_autoencoder


def detect_drift_adwin_online(
        test_X, idx, dist, adwin, autoencoder, encoder, decoder, drift_epochs, drift_batch_size, early_stopping,
        delta_adwin,
        all_losses, final_losses, model=False, test_y=False,
        fit_new_ae_after_drift=False, fit_after_drift=False,
        initialize_new_adwin_after_drift=False, feed_adwin_with_refreshed_re_after_drift=False):
    """

    Parameters
    ----------
    test_X: Data stream with all test data
    idx: Index of outer loop
    dist: List of reconstruction errors of new batch
    adwin: ADWIN change detector
    autoencoder: Trained autoencoder
    encoder: Encoder which belongs to autoencoder
    decoder: Decoder which belongs to autoencoder
    drift_epochs: epochs for training after drift was detected
    drift_batch_size: batch size for training after drift was detected
    early_stopping: determine whether early stopping callback should be used
    delta_adwin: Delta value for ADWIN
    all_losses: Array with losses of each gradient decent
    final_losses: Array with losses of each last training/update epoch
    model: The model if algorithm should be validated with classifier
    test_y: The test labels if algorithm should be validated with classifier
    fit_new_ae_after_drift: Whether a new autoencoder should be trained after drift
    fit_after_drift: Whether autoencoder should be updated by fitting again after drift on ADWIN window
    initialize_new_adwin_after_drift: Whether ADWIN should be initialized after drift
    feed_adwin_with_refreshed_re_after_drift: Whether ADWIN should be refilled after drift

    Returns the widths of ADWIN and the detected drift points and further necessary parameters for algorithm
    -------

    """
    # 1. Initialize arrays
    widths = []
    fusioned_refreshed_dist_adwin = []
    drift_decisions = [False] * len(dist)
    errors_per_dimension = []
    # all_errors_per_dimension = []
    weights_copy = False

    # 2. Adding stream elements to ADWIN and verifying if drift occurred
    for local_idx in range(len(dist)):
        global_idx = idx + local_idx

        # 3. Adding stream elements to ADWIN and verifying if drift occurred
        current_dist = dist[local_idx]
        adwin.add_element(current_dist)
        if adwin.detected_change():
            # 4. Save drift point, error per dimension, and weights of AE
            # Save drift point
            print(f"Change in index {global_idx} for stream value {dist[local_idx]}")
            drift_decisions[local_idx] = True
            # Save reconstruction error per dimension of drift point
            error_per_dimension = compute_error_per_dim(point=global_idx, data_stream_test_x=test_X, encoder=encoder,
                                                        decoder=decoder)
            errors_per_dimension.append(error_per_dimension)
            # Save weights of current autoencoder to detect "where"
            weights_copy = autoencoder.get_weights()

            # 5. Test-then-Train: Define ADWIN window as new train data stream
            window_train_X = test_X[(global_idx - adwin.width): global_idx]
            # 5.1 A new autoencoder should be trained after drift
            if fit_new_ae_after_drift:
                autoencoder, encoder, decoder = build_autoencoder(
                    n_dimensions=autoencoder.input_shape[1], size_encoder=autoencoder.layers[1].output_shape[1])

            # 5.2 Update autoencoder by fitting (again) after drift on ADWIN window
            if fit_after_drift or fit_new_ae_after_drift:
                # Callback will stop training when there is no improvement in loss for three consecutive epochs
                callback = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)] if early_stopping else None
                autoencoder.fit(window_train_X, window_train_X,
                                epochs=drift_epochs,
                                batch_size=drift_batch_size,
                                shuffle=True, verbose=0,
                                callbacks=callback
                                )
                all_losses.extend(autoencoder.history.history['loss'])
                final_losses.append(autoencoder.history.history['loss'][-1])

            # 6. Retrain validation model on ADWIN window
            if model and test_y.any():
                window_train_y = test_y[(global_idx - adwin.width): global_idx]
                model.fit(window_train_X, window_train_y)

            # 7. Calculate refreshed reconstruction error for current ADWIN window
            # Apply updated autoencoder to current ADWIN window
            encoded_refreshed = encoder.predict(window_train_X)
            decoded_refreshed = decoder.predict(encoded_refreshed)
            # Calculate refreshed reconstruction error(s) of elements in current ADWIN window
            refreshed_dist_adwin = np.linalg.norm(window_train_X - decoded_refreshed, axis=1)
            fusioned_refreshed_dist_adwin[-len(refreshed_dist_adwin):] = refreshed_dist_adwin

            # 8. Initialize ADWIN again
            if initialize_new_adwin_after_drift:
                adwin = ADWIN(delta=delta_adwin)
                # 9. Feed ADWIN with refreshed reconstruction errors
                if feed_adwin_with_refreshed_re_after_drift:
                    for i in refreshed_dist_adwin:
                        adwin.add_element(i)

            # 9. Update dist of current tumbling window with refreshed dist
            # Apply updated autoencoder to further elements in tumbling window
            remaining_tw_X = test_X[global_idx:(idx + len(dist))]
            encoded_remaining_tw = encoder.predict(remaining_tw_X)
            decoded_remaining_tw = decoder.predict(encoded_remaining_tw)
            # Calculate refreshed reconstruction error(s) of elements in current ADWIN window
            refreshed_dist_tw = np.linalg.norm(remaining_tw_X - decoded_remaining_tw, axis=1)
            dist[local_idx:] = refreshed_dist_tw

        # Append for every instance (also non-drift) the width of ADWIN and the reconstruction error per dimension
        widths.append(adwin.width)

    return adwin, widths, drift_decisions, errors_per_dimension, weights_copy, fusioned_refreshed_dist_adwin, \
           all_losses, final_losses, autoencoder, encoder, decoder
