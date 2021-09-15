from datetime import datetime
import pickle
import numpy as np
from Datasets.NumpyGenerator.NumpyRandomNormalStreamGenerator import numpy_random_normal_stream_generator
from Datasets.NumpyGenerator.NumpyRandomNormalDriftGenerator import numpy_random_normal_drift_generator


# 0. Set initial parameters
n_dimensions = 100
n_initial_instances = 2000
n_drifts = 10
# Define a minimum of dimensions broken
percentage_min_dimensions_broken = 0.1
min_dimensions_broken = int(percentage_min_dimensions_broken * n_dimensions)
# min_dimensions_broken = 10
# Define a min and a max length of a drift
min_drift_length = 300
max_drift_length = 2000
# a.-c. If drift start and length should be random: Generate indices
drift_indices = np.cumsum(np.ones((n_drifts+1,), np.int) * min_drift_length + np.random.randint(0, max_drift_length, (n_drifts+1,))) - min_drift_length
# # d. If drift start and length should be individualized
# drift_indices = np.array([1000, 3000, 8000])
# Set them after initial instances which do not contain drift
drift_indices = drift_indices + n_initial_instances

# 1. Create file for saving data stream details
date = datetime.today().strftime('%Y-%m-%d_%H.%M')
file_name = 'Generated_Streams/Drift_Labels/RandomNumpyRandomNormalUniform_' \
            + str(n_drifts) + 'DR_' + str(n_dimensions) + 'Dims_' + str(min_dimensions_broken) + 'MinDimBroken_' \
            + str(min_drift_length) + 'MinL_' + str(max_drift_length) + 'MaxL_' + date
file = open(file_name + '.txt', "w")

# 2. Initialize data stream
# Define random initial mean and var
initial_mean = np.random.uniform()
initial_var = np.random.uniform()
# # b. If drift should be only in variance
# initial_mean = 0
# # c. If drift should be only in mean
# initial_var = 0.25
# Add information to txt file
file.write('InitialData' + '_ni' + str(n_initial_instances) + '_nd' + str(n_dimensions)
           + '_m' + str(initial_mean) + '_v' + str(initial_var) + '\n')
# Build complete data stream
data_stream_X = numpy_random_normal_stream_generator(
    mean=initial_mean, var=initial_var, n_instances=max(drift_indices), n_dimensions=n_dimensions)
# Initialize drift labels
drift_labels = np.zeros(len(data_stream_X), dtype=int)

counter = 0
# # d. If drift should be individualized
# drift_means = [2, 4]
# drift_vars = [0.75, 0.5]
# n_dims_broken = [10, 30]
# starts_dims_broken = [40, 60]
for drift_idx in drift_indices[:-1]:
    # Add drift label
    drift_labels[drift_idx] = True
    # Save start point of drifts
    start_instances_broken = drift_idx
    n_drift_instances = drift_indices[counter+1] - drift_indices[counter]

    # # a. Define broken mean and var
    mean_broken = np.random.uniform()
    var_broken = np.random.uniform()
    # # b. If drift should be only in variance
    # mean_broken = initial_mean
    # # c. If drift should be only in mean
    # var_broken = initial_var
    # # d. If drift should be individualized
    # mean_broken = drift_means[counter]
    # var_broken = drift_vars[counter]

    # # a.-c. Define the number of dimensions broken
    n_dimensions_broken = np.random.randint(min_dimensions_broken, n_dimensions + 1)
    start_dimensions_broken = np.random.randint(n_dimensions - n_dimensions_broken + 1)
    # # d. If number of dimensions and start drift should be individualized
    # n_dimensions_broken = n_dims_broken[counter]
    # start_dimensions_broken = starts_dims_broken[counter]

    # Introduce drift in each instance after drift_idx
    affected_stream_part = data_stream_X[drift_idx:]
    data_stream_X[drift_idx:] = numpy_random_normal_drift_generator(
        data_stream=affected_stream_part, mean_broken=mean_broken, var_broken=var_broken,
        n_dimensions_broken=n_dimensions_broken, start_dimensions_broken=start_dimensions_broken)
    # Save information to drift in row of txt file
    file.write('DR' + str(counter+1) + '_i' + str(start_instances_broken) + '_ni' + str(n_drift_instances)
               + '_d' + str(start_dimensions_broken) + '_nd' + str(n_dimensions_broken)
               + '_m' + str(mean_broken) + '_v' + str(var_broken) + '\n')

    counter += 1

# Add drift labels to stream
data_stream = np.append(data_stream_X, drift_labels.reshape(-1, 1), axis=1)

file.close()
# Save data stream to csv
with open(file_name + '.pickle', 'wb') as handle:
    pickle.dump(data_stream, handle, protocol=pickle.HIGHEST_PROTOCOL)

