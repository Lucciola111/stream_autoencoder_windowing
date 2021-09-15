from datetime import datetime
import random
import pickle
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist

# Number of drifts is 19 because we have 10 different digits and 10 different fashion items (first one is start pattern)
n_drifts = 19

# 1. Create file for saving data stream details
date = datetime.today().strftime('%Y-%m-%d_%H.%M')
file_name = 'Generated_Streams/Image_Data_Drift_And_Classifier_Labels/RandomMNIST_and_FashionMNIST_SortAllNumbers' \
            + str(n_drifts) + 'DR_' + date
file = open(file_name + '.txt', "w")

# 2. Initialize data stream
# Load MNIST dataset
(train_mnist_images, train_mnist_labels), (test_mnist_images, test_mnist_labels) = mnist.load_data()
mnist_images = np.concatenate((train_mnist_images, test_mnist_images), axis=0)
mnist_labels = np.concatenate((train_mnist_labels, test_mnist_labels), axis=0)
# Load Fashion-MNIST dataset
(train_fashion_mnist_images, train_fashion_mnist_labels), (test_fashion_mnist_images, test_fashion_mnist_labels) = fashion_mnist.load_data()
fashion_mnist_images = np.concatenate((train_fashion_mnist_images, test_fashion_mnist_images), axis=0)
fashion_mnist_labels = np.concatenate((train_fashion_mnist_labels, test_fashion_mnist_labels), axis=0)
# Create unique labels for MNIST and Fashion-MNIST dataset
fashion_mnist_labels = fashion_mnist_labels + 10
# Merge MNIST and Fashion-MNIST dataset
images = np.concatenate((mnist_images, fashion_mnist_images))
labels = np.concatenate((mnist_labels, fashion_mnist_labels))

# 3. Flatten images
# Define number of instances and dimensions
n_instances = images.shape[0]
n_dimensions = images.shape[1] * images.shape[2]

# Initialize flattened data stream
images_flattened = np.empty([n_instances, n_dimensions])
# Flatten images
index = 0
for x in images:
    images_flattened[index] = x.flatten()
    index += 1

# 4. Form 20 groups of images by label
grouped_images = []
grouped_labels = []
for idx in range(20):
    grouped_images.append(images_flattened[labels == idx])
    grouped_labels.append(labels[labels == idx])
# Randomly shuffle digit and item groups
grouped = list(zip(grouped_images, grouped_labels))
random.shuffle(grouped)
grouped_images, grouped_labels = zip(*grouped)


# 5. Initialization
# Initialize data stream
data_stream_X = grouped_images[0]
data_stream_y = grouped_labels[0]
# Initialize drift labels
drift_labels = np.zeros(len(data_stream_X), dtype=int)

# 6. Append numbers and add drift labels
for idx in range(1, 20):
    # Concatenate streams
    data_stream_X = np.concatenate((data_stream_X, grouped_images[idx]), axis=0)
    data_stream_y = np.concatenate((data_stream_y, grouped_labels[idx]), axis=0)
    # Generate drift labels
    start_instances_broken = len(drift_labels)
    drift_labels = np.append(drift_labels, 1)
    drift_labels = np.append(drift_labels, np.zeros(len(grouped_images[idx]) - 1))
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


