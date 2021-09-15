from datetime import datetime
import pickle
import numpy as np
from tensorflow.keras.datasets import mnist


# 0. Set initial parameters
n_initial_instances = 2000
# Take care with n_drifts, min and max so that partitioning is realistic!!!!
n_drifts = 50
# Define a min and a max length of a drift
min_drift_length = 300
max_drift_length = 2000

# 1. Create file for saving data stream details
date = datetime.today().strftime('%Y-%m-%d_%H.%M')
file_name = 'Generated_Streams/Image_Data_Drift_And_Classifier_Labels/RandomMNIST_VariateTwoSets' \
            + str(n_drifts) + 'DR_' + str(min_drift_length) + 'MinL_' + str(max_drift_length) + 'MaxL_' + date
file = open(file_name + '.txt', "w")

# 2. Initialize data stream
# Load MNIST dataset
(train_mnist_images, train_mnist_labels), (test_mnist_images, test_mnist_labels) = mnist.load_data()
mnist_images = np.concatenate((train_mnist_images, test_mnist_images), axis=0)
mnist_labels = np.concatenate((train_mnist_labels, test_mnist_labels), axis=0)

# 5. Flatten images
# Define number of instances and dimensions
n_instances = mnist_images.shape[0]
n_dimensions = mnist_images.shape[1] * mnist_images.shape[2]
# Initialize flattened data stream
mnist_images_flattened = np.empty([n_instances, n_dimensions])
# Flatten images
index = 0
for x in mnist_images:
    mnist_images_flattened[index] = x.flatten()
    index += 1


# Form two groups of images (0-4 and 5-9)
group1_mnist_images = mnist_images_flattened[mnist_labels < 5]
group1_mnist_labels = mnist_labels[mnist_labels < 5]
group2_mnist_images = mnist_images_flattened[5 <= mnist_labels]
group2_mnist_labels = mnist_labels[5 <= mnist_labels]

# Generate indices for group1
drift_indices_group1 = np.array([])
while drift_indices_group1.size == 0 or max(drift_indices_group1) + n_initial_instances > len(group1_mnist_images):
    drift_indices_group1 = np.cumsum(
        np.ones((int(n_drifts / 2) + 1,), np.int) * min_drift_length + np.random.randint(0, max_drift_length, (int(n_drifts / 2) + 1,))
        ) - min_drift_length
# Set them after initial instances which do not contain drift
# drift_indices_group1 = drift_indices_group1 + n_initial_instances
# Generate indices for group2
drift_indices_group2 = np.array([])
while drift_indices_group2.size == 0 or max(drift_indices_group2) > len(group1_mnist_images):
    drift_indices_group2 = np.cumsum(
        np.ones((int(n_drifts / 2) + 1,), np.int) * min_drift_length + np.random.randint(0, max_drift_length, (int(n_drifts / 2) + 1,))
        ) - min_drift_length
# Combine them always alternating
drift_indices = []
for pair in zip(drift_indices_group2, drift_indices_group1):
    drift_indices.extend(pair)

# Create initial data stream
data_stream_X = group1_mnist_images[:n_initial_instances]
data_stream_y = group1_mnist_labels[:n_initial_instances]

# Initialize drift labels
drift_labels = np.zeros(len(data_stream_X), dtype=int)

counter = 0
for drift_idx in drift_indices[:-2]:

    # Save start point of drifts
    start_instances_broken = drift_idx
    n_drift_instances = drift_indices[counter + 2] - drift_indices[counter]
    if counter % 2 == 0:
        next_stream_part_X = group2_mnist_images[drift_indices[counter] + n_initial_instances:drift_indices[counter] + n_initial_instances + n_drift_instances]
        next_stream_part_y = group2_mnist_labels[drift_indices[counter] + n_initial_instances:drift_indices[counter] + n_initial_instances + n_drift_instances]
    else:
        next_stream_part_X = group1_mnist_images[drift_indices[counter]:drift_indices[counter] + n_drift_instances]
        next_stream_part_y = group1_mnist_labels[drift_indices[counter]:drift_indices[counter] + n_drift_instances]

    # Concatenate streams
    data_stream_X = np.concatenate((data_stream_X, next_stream_part_X), axis=0)
    data_stream_y = np.concatenate((data_stream_y, next_stream_part_y), axis=0)
    # Generate drift labels
    drift_labels = np.append(drift_labels, 1)
    drift_labels = np.append(drift_labels, np.zeros(len(next_stream_part_X) - 1))
    # Add information to txt file
    file.write('DR' + str(drift_idx+1) + '_i' + str(start_instances_broken) + '_ni' + str(n_drift_instances) + '\n')

    counter += 1


# Add class labels and drift labels to stream
data_stream = np.append(data_stream_X, data_stream_y.reshape(-1, 1), axis=1)
data_stream = np.append(data_stream, drift_labels.reshape(-1, 1), axis=1)

file.close()
# Save data stream to csv
with open(file_name + '.pickle', 'wb') as handle:
    pickle.dump(data_stream, handle, protocol=pickle.HIGHEST_PROTOCOL)
