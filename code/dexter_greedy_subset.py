import numpy as np
import scipy.io as io
import scipy.sparse as sp
import math
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from random import shuffle

import lasso

def load_data(X_filename):
    d = 20000
    res = []
    with open(X_filename) as f:
        for line in f:
            cur = np.zeros((d,))
            for token in line.split():
                if len(token) > 1:
                    [feature, val] = token.split(":")
                    cur[int(feature)] = int(val) / 1000.0
            res += [cur]
    return np.array(res)

def load_labels(Y_filename):
    res = []
    with open(Y_filename) as f:
        for line in f:
            if len(line) > 1:
                res += [1 if int(line) > 0 else 0]
    return np.array(res)

def loss01(Y_predict, Y_real):
    return np.sum(Y_predict != Y_real)

def solve_SFG():
    X_train = load_data("dexter_train.data")
    Y_train = load_labels("dexter_train.labels")
    X_valid = load_data("dexter_valid.data")
    Y_valid = load_labels("dexter_valid.labels")
    print "data loaded"

    n, d = X_train.shape
    m, d = X_valid.shape

    # num_iters = 300
    # selected = []

    # for i in range(num_iters):
    #     candidates = [j for j in range(d) if j not in selected]
    #     shuffle(candidates)
    #     accuracy_cancidates = np.zeros((len(candidates),))

    #     for j in range(len(candidates)):
    #         # if j % 1000 == 0:
    #         #     print j,
    #         temp = selected + [candidates[j]]
    #         X_train_selected = X_train[:, temp]
    #         X_valid_selected = X_valid[:, temp]

    #         clf = SVC(kernel="linear")
    #         clf.fit(X_train_selected, Y_train)
    #         prediction = clf.predict(X_valid_selected)
    #         accuracy = 1 - loss01(prediction, Y_valid) * 1.0 / m

    #         clf = SVC(kernel="linear")
    #         clf.fit(X_valid_selected, Y_valid)
    #         prediction = clf.predict(X_train_selected)
    #         accuracy += 1 - loss01(prediction, Y_train) * 1.0 / n
    #         accuracy /= 2

    #         accuracy_cancidates[j] = accuracy
    #         # print accuracy_cancidates[j]

    #     best_next = candidates[np.argmax(accuracy_cancidates)]
    #     selected += [best_next]
    #     print "iter=" + str(i) + "\tloss01=" + str(np.max(accuracy_cancidates))
    #     print selected

    features_added = [10244, 626, 12610, 1565, 8789, 9614, 15798, 3433, 10218, 2062, 2402,
        6084, 8786, 5507, 268, 14973, 2990, 9056, 18413, 370, 14161, 9849, 7596, 6153,
        17942, 7055, 4308, 13685, 4153, 10779, 17471, 13165, 19327, 12666, 10848, 13929,
        12732, 8342, 13863, 4698, 4576, 17173, 17356, 19890, 13881, 16810, 1244, 1987, 3932,
        18115, 4722, 14278, 6554, 8202, 18833, 4857, 10663, 4383, 12196, 16638, 2185, 11140,
        448, 14662, 8707, 12184]

    accuracy2fold = []
    selected = []

    for next_best in features_added:
        selected += [next_best]
        X_train_selected = X_train[:, selected]
        X_valid_selected = X_valid[:, selected]

        clf = SVC(kernel="linear")
        clf.fit(X_train_selected, Y_train)
        prediction = clf.predict(X_valid_selected)
        accuracy = 1 - loss01(prediction, Y_valid) * 1.0 / m

        clf = SVC(kernel="linear")
        clf.fit(X_valid_selected, Y_valid)
        prediction = clf.predict(X_train_selected)
        accuracy += 1 - loss01(prediction, Y_train) * 1.0 / n
        accuracy /= 2

        print accuracy
        accuracy2fold += [accuracy]

    plt.figure()
    plt.plot(range(len(accuracy2fold)), accuracy2fold, '-r')

    plt.ylabel('Accuracy')
    plt.xlabel('k')

    plt.show()


if __name__ == "__main__":
    solve_SFG()
