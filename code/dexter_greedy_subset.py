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
    res = np.array(res)
    return (res - np.mean(res)) / np.std(res)

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
    # 300 * 20000
    # Y labels +1/-1
    X_train = load_data("dexter_train.data")
    Y_train = load_labels("dexter_train.labels")
    X_valid = load_data("dexter_valid.data")
    Y_valid = load_labels("dexter_valid.labels")
    X_valid, X_test = X_valid[0:100], X_valid[100:]
    Y_valid, Y_test = Y_valid[0:100], Y_valid[100:]

    # reorganize to
    # 200 train, 200 valid, 200 test
    X_valid = np.append(X_valid, X_train[0:100], axis=0)
    Y_valid = np.append(Y_valid, Y_train[0:100], axis=0)
    X_train = X_train[100:]
    Y_train = Y_train[100:]
    print "data loaded"

    # n, d = X_train.shape
    # m, d = X_valid.shape

    # num_iters = 70
    # selected = []

    # for i in range(num_iters):
    #     candidates = [j for j in range(d) if j not in selected]
    #     shuffle(candidates)
    #     accuracy_cancidates = np.zeros((len(candidates),))

    #     for j in range(len(candidates)):
    #         if j % 1000 == 0:
    #             print j,
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

    features_added = [12916, 12610, 3433, 17104, 19674, 14844, 4637, 3870, 7577,
        18201, 2851, 17356, 5435, 18224, 9734, 16511, 5801, 4074, 3147, 15937,
        7651, 2982, 14994, 9419, 16509, 4325, 15629, 14299, 5075, 13587, 12552,
        4644, 17228, 14645, 3313, 17812, 8361, 3814, 12233, 11324, 5575, 18976,
        2472, 12042, 5658, 7652, 14573, 14892, 18801, 11188, 12026, 12245, 9494,
        17541, 17700, 5549, 12311, 19863, 13398, 16561, 1058, 2368, 619, 10539,
        17484, 5203, 17080, 19109, 10292, 4857]
    # features_added = selected

    accuracies = []
    selected = []

    for next_best in features_added:
        selected += [next_best]
        X_train_selected = X_train[:, selected]
        X_test_selected = X_test[:, selected]

        clf = SVC(kernel="linear")
        clf.fit(X_train_selected, Y_train)
        prediction = clf.predict(X_test_selected)
        accuracy = 1 - loss01(prediction, Y_test) * 1.0 / Y_test.shape[0]

        accuracies += [accuracy]
        

    print accuracies

    plt.figure()
    plt.plot(np.array(range(len(accuracies))) + 1, accuracies, '-r')

    plt.ylabel('Accuracy')
    plt.xlabel('k')

    plt.show()


if __name__ == "__main__":
    solve_SFG()
