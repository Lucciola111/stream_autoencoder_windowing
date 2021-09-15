from Datasets.NumpyGenerator.NumpyRandomNormalStreamGenerator import numpy_random_normal_stream_generator


def numpy_random_normal_drift_generator(
        data_stream, mean_broken, var_broken, n_dimensions_broken, start_dimensions_broken):
    """

    Parameters
    ----------
    data_stream: data stream where drift should be introduced
    mean_broken: the mean of the drift
    var_broken: the standard deviation of the drift
    n_dimensions_broken: number of affected dimensions
    start_dimensions_broken: start dimension for broken dimensions

    Returns a data stream with sudden drift in some dimensions

    """
    n_instances = len(data_stream)

    # Generate data with broken mean and var
    dimensions_broken = numpy_random_normal_stream_generator(
             mean=mean_broken, var=var_broken, n_instances=n_instances, n_dimensions=n_dimensions_broken)

    # Replace selected dimensions with generated drift dimensions
    data_stream[:, start_dimensions_broken:(start_dimensions_broken + n_dimensions_broken)] = dimensions_broken

    return data_stream
