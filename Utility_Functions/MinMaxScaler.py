import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


def min_max_scaler(data_stream_X, columnwise=False):

    if not columnwise:
        # Calculate min and max of whole data stream
        min_val = data_stream_X.min()
        max_val = data_stream_X.max()
        # Normalize data to [0,1]
        data_stream_X = (data_stream_X - min_val) / (max_val - min_val)

    else:
        # Normalize each dimension of the dataset individually to [0,1]
        min_max_scaler_columnwise = MinMaxScaler()
        min_max_scaler.fit(data_stream_X)
        data_stream_X = min_max_scaler_columnwise.transform(data_stream_X)

    return data_stream_X
