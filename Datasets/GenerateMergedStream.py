import pickle
import numpy as np
from TrainingFunctions.LoadDataStream import load_data_stream

separate_train_test_file = False
image_data = False
drift_labels_known = True

min_drift_distance = 300

path_A = "Generated_Streams/Drift_Labels/"
dataset_A = "RandomNumpyRandomNormalUniform_50DR_100Dims_1MinDimBroken_300MinL_2000MaxL_2021-08-06_10.53.pickle"
path_B = "Generated_Streams/Drift_And_Classifier_Labels/"
dataset_B = "RandomRandomRBF_50DR_100Dims_50Centroids_1MinDriftCentroids_300MinL_2000MaxL_2021-08-06_10.57.pickle"

# Load data stream A
proxy_evaluation_A = False
data_stream_A = load_data_stream(dataset=dataset_A, path=path_A, separate_train_test_file=separate_train_test_file,
                                 image_data=image_data, drift_labels_known=drift_labels_known,
                                 proxy_evaluation=proxy_evaluation_A)
# Load data stream B
proxy_evaluation_B = True
data_stream_B = load_data_stream(dataset=dataset_B, path=path_B, separate_train_test_file=separate_train_test_file,
                                 image_data=image_data, drift_labels_known=drift_labels_known,
                                 proxy_evaluation=proxy_evaluation_B)
# Separate drift labels and data stream
if drift_labels_known:
    data_stream_A, drift_labels_A = data_stream_A[:, :-1], data_stream_A[:, -1]
    drift_points_A = list(np.where(drift_labels_A)[0])
    data_stream_B, drift_labels_B = data_stream_B[:, :-1], data_stream_B[:, -1]
    drift_points_B = list(np.where(drift_labels_B)[0])
actual_drift_points = sorted(np.concatenate((drift_points_A, drift_points_B)))

# Remove class labels
if proxy_evaluation_A:
    data_stream_A = data_stream_A[:, :-1]
if proxy_evaluation_B:
    data_stream_B = data_stream_B[:, :-1]

# Merge data streams so that minimum distance between drifts is kept
for idx in range(len(actual_drift_points)):
    if idx < len(actual_drift_points) and actual_drift_points[idx] < actual_drift_points[-1] and (actual_drift_points[idx+1] - actual_drift_points[idx]) < min_drift_distance:
        if actual_drift_points[idx] in drift_points_B:
            # Define elements to delete so that drift point is the same as next drift in B
            idx_to_delete = list(range(actual_drift_points[idx], actual_drift_points[idx+1]))
            # Delete instances and labels of data stream A
            data_stream_A = np.delete(data_stream_A, idx_to_delete, axis=0)
            drift_labels_A = np.delete(drift_labels_A, idx_to_delete, axis=0)

        if actual_drift_points[idx] in drift_points_A:
            length_pattern = actual_drift_points[idx+1] - actual_drift_points[idx]
            # Generate new pattern from previous data with the length until next drift in B
            new_pattern_stream = data_stream_A[actual_drift_points[idx]-length_pattern:actual_drift_points[idx]]
            new_labels_stream = np.zeros((len(new_pattern_stream)), dtype=bool)
            # Insert new pattern and labels of data stream A
            data_stream_A = np.insert(data_stream_A, actual_drift_points[idx], new_pattern_stream, axis=0)
            drift_labels_A = np.insert(drift_labels_A, actual_drift_points[idx], new_labels_stream, axis=0)

        # Update drift labels
        drift_points_A = list(np.where(drift_labels_A)[0])
        actual_drift_points = np.unique(sorted(np.concatenate((drift_points_A, drift_points_B))), axis=0)

# Cut streams to same length
if len(data_stream_A) > len(data_stream_B):
    data_stream_A = data_stream_A[:len(data_stream_B)]
    drift_labels_A = drift_labels_A[:len(drift_labels_B)]
else:
    data_stream_B = data_stream_B[:len(data_stream_A)]
    drift_labels_B = drift_labels_B[:len(drift_labels_A)]

data_stream_mixed = np.concatenate((data_stream_A, data_stream_B), axis=1)
drift_labels_mixed = np.logical_or(drift_labels_A, drift_labels_B)

# Add drift labels to stream
data_stream = np.append(data_stream_mixed, drift_labels_mixed.reshape(-1, 1), axis=1)


cut_A = dataset_A.index('MinL_')
cut_B = dataset_B.index('MinL_')
file_name_A = dataset_A[:(cut_A-4)]
file_name_B = dataset_B[:(cut_B-4)]

# 1. Create file for saving data stream details
file_name = 'Generated_Streams/Drift_Labels/Mixed_' + str(min_drift_distance) + 'MinDistance_' \
            + 'DATASET_A_' + str(file_name_A) + '_DATASET_B_' + str(file_name_B)
# Save data stream to csv
with open(file_name + '.pickle', 'wb') as handle:
    pickle.dump(data_stream, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Verification:
# Returns minimum difference between any pair
def find_min_diff(array, n):
    # Sort array in non-decreasing order
    array = sorted(array)
    # Initialize difference as infinite
    diff = 10 ** 20
    # Find the min diff by comparing adjacent pairs in sorted array
    for i in range(n - 1):
        if array[i + 1] - array[i] < diff:
            diff = array[i + 1] - array[i]
    # Return min diff
    return diff

n = len(actual_drift_points)
min_diff = find_min_diff(array=actual_drift_points, n=n)

