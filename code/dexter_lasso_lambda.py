import numpy as np
import scipy.io as io
import scipy.sparse as sp
import math
from sklearn.svm import SVC
import matplotlib.pyplot as plt

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

def lasso_predict(w_hat, w_0_hat, X):
    res = X.dot(w_hat) + w_0_hat
    return np.array([1 if res[i] > 0.5 else 0 for i in range(res.shape[0])])

def loss01(Y_predict, Y_real):
    return np.sum(Y_predict != Y_real)

def accuracy(Y_predict, Y_real):
    return 1 - loss01(Y_predict, Y_real) * 1.0 / Y_real.shape[0]

def solve():
    X_train = sp.csc_matrix(load_data("dexter_train.data"))
    Y_train = load_labels("dexter_train.labels")
    X_valid = sp.csc_matrix(load_data("dexter_valid.data"))
    Y_valid = load_labels("dexter_valid.labels")

    X_valid, X_test = X_valid[0:100], X_valid[100:]
    Y_valid, Y_test = Y_valid[0:100], Y_valid[100:]

    print "data loaded"

    lamb = 0.261673816514
    num_nonzeros = 0
    dims = []
    accuracies = []

    while (num_nonzeros <= 300):
        w_hat, w_0_hat = lasso.solve_lasso_with_lamb(X_train, Y_train, lamb)
        num_nonzeros = lasso.num_nonzeros(w_hat)
        print "lamb=" + str(lamb) + "\t num nonzeros in w = " + str(num_nonzeros)

        selected = []
        for i in range(len(w_hat)):
            if (abs(w_hat[i]) > 1e-09):
                selected += [i]
        print selected[0:20]

        X_train_selected = X_train[:, selected]
        X_test_selected = X_test[:, selected]

        clf = SVC(kernel="linear")
        clf.fit(X_train_selected, Y_train)

        prediction = clf.predict(X_train_selected)
        accuracy_train = 1 - loss01(prediction, Y_train) * 1.0 / Y_train.shape[0]
        print "  train_accuracy=" + str(accuracy_train)

        prediction = clf.predict(X_test_selected)
        accuracy = 1 - loss01(prediction, Y_test) * 1.0 / Y_test.shape[0]

        dims += [num_nonzeros]
        accuracies += [accuracy]
        print "  test_accuracy=" + str(accuracy)

        lamb *= .9


    plt.figure()
    plt.plot(dims, accuracies, '-r')

    plt.ylabel('Accuracy')
    plt.xlabel('k')

    plt.show()

if __name__ == "__main__":
    solve()
