from Utility_Functions.MinMaxScaler import min_max_scaler


def preprocessing_with_labels(data_stream, n_instances, n_train_data, n_val_data, n_test_data, image_data=False):

    # Separate features and labels
    data_stream_X, data_stream_y = data_stream[:, :-1], data_stream[:, -1]

    if image_data:
        # Scale values to a range from 0 to 1
        data_stream_X = data_stream_X.astype('float32') / 255.
    else:
        # Normalize data to [0,1]
        data_stream_X = min_max_scaler(data_stream_X)

    # Split features in training, test and validation data
    train_X, train_y = data_stream_X[0:n_train_data], data_stream_y[0:n_train_data]
    val_X, val_y = data_stream_X[n_train_data:(n_train_data + n_val_data)], data_stream_y[n_train_data:(n_train_data + n_val_data)]
    test_X, test_y = data_stream_X[(n_instances - n_test_data):n_instances], data_stream_y[(n_instances - n_test_data):n_instances]

    return train_X, train_y, val_X, val_y, test_X, test_y
