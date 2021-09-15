import numpy as np


def numpy_random_normal_stream_generator(mean, var, n_instances, n_dimensions):
    """

    Parameters
    ----------
    mean: mean for generated data stream
    var: standard deviation for generated data stream
    n_instances: number of rows
    n_dimensions: number of columns

    Returns a data stream with a normal distribution
    -------

    """

    # Generate multi-dimensional data
    means = np.asarray([mean] * n_instances * n_dimensions)  # generate mean for every data point
    means = means.reshape(n_instances, n_dimensions)  # make it multidimensional with shape n_instances, n_dimensions
    variances = np.asarray([var] * n_instances * n_dimensions)  # generate standard deviation for every data point
    variances = variances.reshape(n_instances, n_dimensions)  # make it multidimensional with shape n_instances, n_dimensions

    data_stream = np.random.normal(means, variances)

    return data_stream
