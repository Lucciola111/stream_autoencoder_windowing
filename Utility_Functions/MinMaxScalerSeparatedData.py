from sklearn.preprocessing import MinMaxScaler


def min_max_scaler_separated_data(data_stream, train_X, val_X, test_X, columnwise=False):

    if not columnwise:
        # Calculate min and max of whole data stream
        min_val = data_stream[:, :-1].min()
        max_val = data_stream[:, :-1].max()
        # Normalize data to [0,1]
        train_X = (train_X - min_val) / (max_val - min_val)
        val_X = (val_X - min_val) / (max_val - min_val)
        test_X = (test_X - min_val) / (max_val - min_val)

    else:
        # Normalize each dimension of the dataset individually to [0,1]
        min_max_scaler_columnwise = MinMaxScaler()
        min_max_scaler_columnwise.fit(data_stream[:, :-1])
        train_X = min_max_scaler_columnwise.transform(train_X)
        val_X = min_max_scaler_columnwise.transform(val_X)
        test_X = min_max_scaler_columnwise.transform(test_X)

    return train_X, val_X, test_X
