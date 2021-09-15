from datetime import datetime
import pickle
import numpy as np
from skmultiflow.data.random_rbf_generator import RandomRBFGenerator
from skmultiflow.data.random_rbf_generator_drift import RandomRBFGeneratorDrift


# 0. Set initial parameters
n_dimensions = 100
n_initial_instances = 2000
n_drifts = 50
n_classes = 2
n_centroids = 50
# Define a minimum of drifting centroids
min_drift_centroids = 1
# Define a min and a max length of a drift
min_drift_length = 300
max_drift_length = 2000
# Generate indices
drift_indices = np.cumsum(np.ones((n_drifts+1,), np.int) * min_drift_length + np.random.randint(0, max_drift_length-min_drift_length+1, (n_drifts+1,))) - min_drift_length
# Set them after initial instances which do not contain drift
drift_indices = drift_indices + n_initial_instances

# 1. Create file for saving data stream details
date = datetime.today().strftime('%Y-%m-%d_%H.%M')
file_name = 'Generated_Streams/Drift_And_Classifier_Labels/RandomRandomRBF_' \
            + str(n_drifts) + 'DR_' + str(n_dimensions) + 'Dims_' + str(n_centroids) + 'Centroids_' \
            + str(min_drift_centroids) + 'MinDriftCentroids_' \
            + str(min_drift_length) + 'MinL_' + str(max_drift_length) + 'MaxL_' + date
file = open(file_name + '.txt', "w")

# 2. Initialize data stream
# Generate normal part of data stream
data_stream_normal = RandomRBFGenerator(model_random_state=99, sample_random_state=50, n_classes=n_classes,
                                        n_features=n_dimensions, n_centroids=n_centroids)
# Build complete data stream
data_stream_X, data_stream_y = data_stream_normal.next_sample(max(drift_indices))

# Add information to txt file
file.write('InitialData' + '_ni' + str(n_initial_instances) + '_nd' + str(n_dimensions)
           + '_nclasses' + str(n_classes) + '_ncentroids' + str(n_centroids) + '\n')

# Initialize drift labels
drift_labels = np.zeros(len(data_stream_X), dtype=int)

# change_speeds = [0.5, 0.001, 0.01, 0.1, 0.5, 0.87, 1]

counter = 0
for drift_idx in drift_indices[:-1]:
    print(counter)
    # Add drift label
    drift_labels[drift_idx] = True
    # Save start point of drifts
    start_instances_broken = drift_idx
    n_drift_instances = drift_indices[counter+1] - drift_indices[counter]

    # Define change speed
    change_speed = np.random.uniform()
    # change_speed = change_speeds[counter]

    # Define the number of drift centroids
    n_drift_centroids = np.random.randint(min_drift_centroids, n_centroids + 1)
    # Introduce drift in each instance after drift_idx
    affected_stream_part_X = data_stream_X[drift_idx:]
    affected_stream_part_y = data_stream_y[drift_idx:]
    # Generate data stream with gradual drift
    data_stream_drift = RandomRBFGeneratorDrift(model_random_state=99, sample_random_state=50, n_classes=n_classes,
                                                n_features=n_dimensions, n_centroids=n_centroids,
                                                change_speed=change_speed, num_drift_centroids=n_drift_centroids)
    remaining_stream_elements = len(affected_stream_part_X)
    data_stream_drift_X, data_stream_drift_y = data_stream_drift.next_sample(remaining_stream_elements)
    # Replace data stream with drift instances
    data_stream_X[drift_idx:] = data_stream_drift_X
    data_stream_y[drift_idx:] = data_stream_drift_y

    # Save information to drift in row of txt file
    file.write('DR' + str(counter+1) + '_i' + str(start_instances_broken) + '_ni' + str(n_drift_instances)
               + '_cs' + str(change_speed) + '_ndc' + str(n_drift_centroids) + '\n')

    counter += 1

# Add class labels to stream
data_stream = np.append(data_stream_X, data_stream_y.reshape(-1,1), axis=1)
# Add drift labels to stream
data_stream = np.append(data_stream, drift_labels.reshape(-1, 1), axis=1)

file.close()
# Save data stream to csv
with open(file_name + '.pickle', 'wb') as handle:
    pickle.dump(data_stream, handle, protocol=pickle.HIGHEST_PROTOCOL)


