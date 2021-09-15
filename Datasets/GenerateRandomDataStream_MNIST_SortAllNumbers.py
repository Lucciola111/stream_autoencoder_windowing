from datetime import datetime
import pickle
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist

# Number of drifts is 9 because we have 10 different digits
n_drifts = 9

# 1. Create file for saving data stream details
date = datetime.today().strftime('%Y-%m-%d_%H.%M')
file_name = 'Generated_Streams/Image_Data_Drift_And_Classifier_Labels/RandomMNIST_SortAllNumbers' \
            + str(n_drifts) + 'DR_' + date
file = open(file_name + '.txt', "w")

# 2. Initialize data stream
# Load MNIST dataset
(train_mnist_images, train_mnist_labels), (test_mnist_images, test_mnist_labels) = mnist.load_data()
mnist_images = np.concatenate((train_mnist_images, test_mnist_images), axis=0)
mnist_labels = np.concatenate((train_mnist_labels, test_mnist_labels), axis=0)

# 3. Flatten images
# Define number of instances and dimensions
n_mnist_instances = mnist_images.shape[0]
n_mnist_dimensions = mnist_images.shape[1] * mnist_images.shape[2]
# Initialize flattened data stream
mnist_images_flattened = np.empty([n_mnist_instances, n_mnist_dimensions])
# Flatten images
index = 0
for x in mnist_images:
    mnist_images_flattened[index] = x.flatten()
    index += 1

# 4. Form 10 groups of images by label
grouped_mnist_images = []
grouped_mnist_labels = []
for idx in range(10):
    grouped_mnist_images.append(mnist_images_flattened[mnist_labels == idx])
    grouped_mnist_labels.append(mnist_labels[mnist_labels == idx])

# 5. Initialization
# Initialize data stream
data_stream_X = grouped_mnist_images[0]
data_stream_y = grouped_mnist_labels[0]
# Initialize drift labels
drift_labels = np.zeros(len(data_stream_X), dtype=int)

# 6. Append numbers and add drift labels
for idx in range(1, 10):
    # Concatenate streams
    data_stream_X = np.concatenate((data_stream_X, grouped_mnist_images[idx]), axis=0)
    data_stream_y = np.concatenate((data_stream_y, grouped_mnist_labels[idx]), axis=0)
    # Generate drift labels
    start_instances_broken = len(drift_labels)
    drift_labels = np.append(drift_labels, 1)
    drift_labels = np.append(drift_labels, np.zeros(len(grouped_mnist_images[idx]) - 1))
    n_drift_instances = len(drift_labels) - start_instances_broken

    # Add information to txt file
    file.write('DR' + str(idx) + '_i' + str(start_instances_broken) + '_ni' + str(n_drift_instances) + '\n')

file.close()

# 7. Add class labels and drift labels to stream
data_stream = np.append(data_stream_X, data_stream_y.reshape(-1, 1), axis=1)
data_stream = np.append(data_stream, drift_labels.reshape(-1, 1), axis=1)

# 8. Save data stream to csv
with open(file_name + '.pickle', 'wb') as handle:
    pickle.dump(data_stream, handle, protocol=pickle.HIGHEST_PROTOCOL)


