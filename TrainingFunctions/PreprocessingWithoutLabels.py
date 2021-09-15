from Utility_Functions.MinMaxScaler import min_max_scaler


def preprocessing_without_labels(data_stream, n_instances, n_train_data, n_val_data, n_test_data):
    # Normalize data to [0,1]
    data_stream_X = min_max_scaler(data_stream)
    # Split features in training, test and validation data
    train_X = data_stream_X[0:n_train_data]
    val_X = data_stream_X[n_train_data:(n_train_data + n_val_data)]
    test_X = data_stream_X[(n_instances - n_test_data):n_instances]

    return train_X, val_X, test_X
