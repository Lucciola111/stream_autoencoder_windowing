from time import process_time as timer
import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.metrics import mean_squared_error
from random import seed, shuffle
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import matplotlib.pyplot as plt


def IBDD(train_X, test_X, window_length, consecutive_values, train_y=False, test_y=False):
    files2del = ['w1.jpeg', 'w2.jpeg', 'w1_cv.jpeg', 'w2_cv.jpeg']

    proxy_evaluation = True if (type(train_y) is pd.Series and type(test_y) is pd.Series) else False
    drift_decisions = [False] * len(test_X)

    n_runs = 20
    if proxy_evaluation:
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
        model.fit(train_X, train_y)

    if window_length > len(train_X):
        window_length = len(train_X)

    superior_threshold, inferior_threshold, nrmse = find_initial_threshold(train_X, window_length, n_runs)
    threshold_diffs = [superior_threshold - inferior_threshold]

    recent_data_X = train_X.iloc[-window_length:].copy()
    if proxy_evaluation:
        recent_data_y = train_y.iloc[-window_length:].copy()

    w1 = get_imgdistribution("w1.jpeg", recent_data_X)
    vet_acc = np.zeros(len(test_X))
    lastupdate = 0
    start = timer()
    print('IBDD Running...')
    for i in range(0, len(test_X)):
        print('Example {}/{}'.format(i + 1, len(test_X)), end='\r')
        if proxy_evaluation:
            prediction = model.predict(test_X.iloc[[i]])
            if prediction == test_y[i]:
                vet_acc[i] = 1

        recent_data_X.drop(recent_data_X.index[0], inplace=True, axis=0)
        recent_data_X = recent_data_X.append(test_X.iloc[[i]], ignore_index=True)
        if proxy_evaluation:
            recent_data_y.drop(recent_data_y.index[0], inplace=True, axis=0)
            recent_data_y = recent_data_y.append(test_y.iloc[[i]], ignore_index=True)

        w2 = get_imgdistribution("w2.jpeg", recent_data_X)

        nrmse.append(mean_squared_error(w1, w2))

        if i - lastupdate > 60:
            superior_threshold = np.mean(nrmse[-50:]) + 2 * np.std(nrmse[-50:])
            inferior_threshold = np.mean(nrmse[-50:]) - 2 * np.std(nrmse[-50:])
            threshold_diffs.append(superior_threshold - inferior_threshold)
            lastupdate = i

        if all(i >= superior_threshold for i in nrmse[-consecutive_values:]):
            superior_threshold = nrmse[-1] + np.std(nrmse[-50:-1])
            inferior_threshold = nrmse[-1] - np.mean(threshold_diffs)
            threshold_diffs.append(superior_threshold - inferior_threshold)
            drift_decisions[i] = True
            if proxy_evaluation:
                model.fit(recent_data_X, recent_data_y)
            lastupdate = i

        elif all(i <= inferior_threshold for i in nrmse[-consecutive_values:]):
            inferior_threshold = nrmse[-1] - np.std(nrmse[-50:-1])
            superior_threshold = nrmse[-1] + np.mean(threshold_diffs)
            threshold_diffs.append(superior_threshold - inferior_threshold)
            drift_decisions[i] = True
            if proxy_evaluation:
                model.fit(recent_data_X, recent_data_y)
            lastupdate = i

    end = timer()
    execution_time = end - start
    mean_acc = np.mean(vet_acc) * 100
    print('\nFinished!')
    print('{} drifts detected at {}'.format(sum(drift_decisions), list(np.where(drift_decisions)[0])))
    print('Average classification accuracy: {}%'.format(np.round(mean_acc, 2)))
    print('Time per example: {} sec'.format(np.round(execution_time / len(test_X), 2)))
    print('Total time: {} sec'.format(np.round(execution_time, 2)))

    for f in files2del:
        os.remove(f)
    return drift_decisions, vet_acc, mean_acc, execution_time


def find_initial_threshold(X_train, window_length, n_runs):
    if window_length > len(X_train):
        window_length = len(X_train)

    w1 = X_train.iloc[-window_length:].copy()
    w1_cv = get_imgdistribution("w1_cv.jpeg", w1)

    max_index = X_train.shape[0]
    sequence = [i for i in range(max_index)]
    nrmse_cv = []
    for i in range(0, n_runs):
        # seed random number generator
        seed(i)
        # randomly shuffle the sequence
        shuffle(sequence)
        w2 = X_train.iloc[sequence[:window_length]].copy()
        w2.reset_index(drop=True, inplace=True)
        w2_cv = get_imgdistribution("w2_cv.jpeg", w2)
        nrmse_cv.append(mean_squared_error(w1_cv, w2_cv))
        threshold1 = np.mean(nrmse_cv) + 2 * np.std(nrmse_cv)
        threshold2 = np.mean(nrmse_cv) - 2 * np.std(nrmse_cv)
    if threshold2 < 0:
        threshold2 = 0
    return threshold1, threshold2, nrmse_cv


def get_imgdistribution(name_file, data):
    plt.imsave(name_file, data.transpose(), cmap='Greys', dpi=100)
    w = imread(name_file)
    return w


def wrs_test(train_data, test_data, window_length, threshold):

    train_X = train_data.iloc[:, :-1]
    test_X = test_data.iloc[:, :-1]
    train_y = train_data.iloc[:, -1]
    test_y = test_data.iloc[:, -1]

    if window_length > len(train_y):
        window_length = len(train_y)

    print('WRS Running...')
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
    model.fit(train_X, train_y)

    w1 = train_X.iloc[-window_length:].copy()
    w2 = train_X.iloc[-window_length:].copy()
    w2_labels = train_y.iloc[-window_length:].copy()

    vet_acc = np.zeros(len(test_y))
    _, n_features = test_X.shape
    drift_points = []
    flag = False
    start = timer()
    for i in range(0, len(test_X)):
        print('Example {}/{}'.format(i + 1, len(test_y)), end='\r')
        prediction = model.predict(test_X.iloc[[i]])
        if prediction == test_y[i]:
            vet_acc[i] = 1
        w2.drop(w2.index[0], inplace=True, axis=0)
        w2 = w2.append(test_X.iloc[[i]], ignore_index=True)
        w2_labels.drop(w2_labels.index[0], inplace=True, axis=0)
        w2_labels = w2_labels.append(test_y.iloc[[i]], ignore_index=True)

        # statistical test for each feature
        for j in range(0, n_features):
            _, p_value = stats.ranksums(w1.iloc[:, j], w2.iloc[:, j])
            if p_value <= threshold:
                flag = True

        if flag:
            drift_points.append(i)
            w1 = w2  # update the reference window with recent data of w2
            model.fit(w2, w2_labels)  # update the classification model with recent data
            flag = False

    end = timer()
    execution_time = end - start
    mean_acc = np.mean(vet_acc) * 100
    print('\nFinished!')
    print('{} drifts detected at {}'.format(len(drift_points), drift_points))
    print('Average classification accuracy: {}%'.format(np.round(mean_acc, 2)))
    print('Time per example: {} sec'.format(np.round(execution_time / len(test_y), 2)))
    print('Total time: {} sec'.format(np.round(execution_time, 2)))

    plot_acc(vet_acc, 500, '', 'dashed', 'WRS')
    return drift_points, vet_acc, mean_acc, execution_time


def baseline_classifier(train_data, test_data):

    train_X = train_data.iloc[:, :-1]
    test_X = test_data.iloc[:, :-1]
    train_y = train_data.iloc[:, -1]
    test_y = test_data.iloc[:, -1]

    vet_acc = np.zeros(len(test_y))
    print('Baseline Running...')
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
    model.fit(train_X, train_y)
    start = timer()
    for i in range(0, len(test_y)):
        print('Example {}/{}'.format(i + 1, len(test_y)), end='\r')
        prediction = model.predict(test_X.iloc[[i]])
        if prediction == test_y[i]:
            vet_acc[i] = 1
    end = timer()
    print('\nFinished!')
    execution_time = end - start
    mean_acc = np.mean(vet_acc) * 100
    print('Average classification accuracy: {}%'.format(np.round(mean_acc, 2)))
    print('Time per example: {} sec'.format(np.round(execution_time / len(test_y), 2)))
    print('Total time: {} sec'.format(np.round(execution_time, 2)))
    plot_acc(vet_acc, 500, 's', '-', 'Baseline')
    return mean_acc, vet_acc, execution_time


def topline_classifier(train_data, test_data, window_length):

    train_X = train_data.iloc[:, :-1]
    test_X = test_data.iloc[:, :-1]
    train_y = train_data.iloc[:, -1]
    test_y = test_data.iloc[:, -1]

    vet_acc = np.zeros(len(test_y))
    print('Topline Running...')
    if window_length > len(train_y):
        window_length = len(train_y)
    start = timer()
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
    model.fit(train_X, train_y)
    start = timer()
    for i in range(0, window_length):
        print('Example {}/{}'.format(i + 1, len(test_y)), end='\r')
        prediction = model.predict(test_X.iloc[[i]])
        if prediction == test_y[i]:
            vet_acc[i] = 1

    for i in range(window_length, len(test_y)):
        print('Example {}/{}'.format(i + 1, len(test_y)), end='\r')
        model.fit(test_X.iloc[i - window_length:i], test_y.iloc[i - window_length:i])
        prediction = model.predict(test_X.iloc[[i]])
        if prediction == test_y[i]:
            vet_acc[i] = 1
    end = timer()
    print('Finished!')
    execution_time = end - start
    mean_acc = np.mean(vet_acc) * 100
    print('\nFinished!')
    print('Average classification accuracy: {}%'.format(np.round(mean_acc, 2)))
    print('Time per example: {} sec'.format(np.round(execution_time / len(test_y), 2)))
    print('Total time: {} sec'.format(np.round(execution_time, 2)))
    plot_acc(vet_acc, 500, '^', '-', 'Topline')
    return mean_acc, vet_acc, execution_time


def plot_acc(vet_acc, window, marker_type, line, method_name, drifts=False):
    vet_len = len(vet_acc)
    mean_acc = []
    for i in range(0, vet_len, window):
        mean_acc.append(np.mean(vet_acc[i:i + window]))

    fig, ax = plt.subplots(figsize=(4, 2))
    plt.plot([float(x) * window for x in range(0, len(mean_acc))], mean_acc, marker=marker_type, ls=line,
             label=method_name)

    if drifts:
        for xc in drifts:
            plt.axvline(x=xc, c='red')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('Examples')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
