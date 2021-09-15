import numpy as np


def compute_error_per_dim(point, data_stream_test_x, encoder, decoder):
    """

    Parameters
    ----------
    point: Examined data point
    data_stream_test_x: Data stream
    encoder: Encoder
    decoder: Decoder

    Returns Array with reconstruction error for each dimension of a data point
    -------

    """
    n_dimensions = data_stream_test_x.shape[1]
    p = np.array(data_stream_test_x[point, :]).reshape(1, n_dimensions)
    encoded = encoder.predict(p)
    decoded = decoder.predict(encoded)
    return np.array(abs(p - decoded))[0]
