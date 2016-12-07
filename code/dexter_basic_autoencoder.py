import numpy as np
import scipy.io as io
import scipy.sparse as sp
import math
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import random

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
                res += [1 if int(line) > 0 else -1]
    return np.array(res)

def loss01(Y_predict, Y_real):
    return np.sum(Y_predict != Y_real)

def accuracy(Y_predict, Y_real):
    return 1 - loss01(Y_predict, Y_real) * 1.0 / Y_real.shape[0]

# def relu(V):
#     return np.maximum(V, 0)

# def sigmoid(V):
#     return 1.0 / (1 + np.exp(-V))

# def d_relu(V):
#     return np.minimum(np.maximum(V, 0.), 1.)

# def d_tanh(V):
#     temp = np.tanh(V)
#     return 1. - np.multiply(temp, temp)

# def d_sigmoid(V):
#     temp = sigmoid(V)
#     return np.multiply(temp, 1 - temp)

def square_loss(P, Y_actual):
    # print P[0][0:3]
    # print Y_actual[0][0:3]
    temp = P - Y_actual
    loss = np.sum(np.sqrt(np.multiply(temp, temp)))
    return loss * 1. / Y_actual.shape[0]

def train_2_layer_autoencoder(X_train, dim_size, batch_size=10):
    n, original_d = X_train.shape

    shuffled_indexes = range(n)
    random.shuffle(shuffled_indexes)
    batch_indexes = [shuffled_indexes[i:i + batch_size] for i in xrange(0, n, batch_size)]
    X_batches = np.array([X_train[xs] for xs in batch_indexes])
    Y_batches = np.array([X_train[xs] for xs in batch_indexes])
    
    # Initialize weights to small random values
    W_xh = np.array([ [np.random.normal(scale=0.0001) for j in range(dim_size)] for i in range(original_d) ])
    W_hp = np.array([ [np.random.normal(scale=0.0001) for j in range(original_d)] for i in range(dim_size) ])
    b_h = np.zeros((dim_size,))
    b_p = np.zeros((original_d,))

    learning_rate = 0.01

    cur_batch = 0
    epoch = 0
    max_epoch = 5
    num_batches_per_epoch = n / batch_size
 
    while (epoch < max_epoch):
        X_batch = X_batches[cur_batch]
        Y_batch = Y_batches[cur_batch]

        # forward
        H = X_batch.dot(W_xh) + b_h
        P = H.dot(W_hp) + b_p
        E = square_loss(P, Y_batch)
        # print E

        # backprop
        dE_dP = 2. / batch_size * (P - Y_batch)
        # print "dE_dP: " + str(dE_dP[0][0:3])
        dE_dH = dE_dP.dot(W_hp.T)
        # print "dE_dH: " + str(dE_dH.shape)
        dE_dW_hp = H.T.dot(dE_dP)
        # print "dE_dW_hp: " + str(dE_dW_hp[0][0:5])
        dE_db_p = np.ones((batch_size)).dot(dE_dP)
        # print "dE_db_p: " + str(dE_db_p.shape)
        dE_dW_xh = X_batch.T.dot(dE_dH)
        # print "dE_dW_xh: " + str(dE_dW_xh[0][0:5])
        dE_db_h = np.ones((batch_size)).dot(dE_dH)
        # print "dE_db_h: " + str(dE_db_h.shape)

        W_hp -= learning_rate * dE_dW_hp
        b_p -= learning_rate * dE_db_p
        W_xh -= learning_rate * dE_dW_xh
        b_h -= learning_rate * dE_db_h

        if cur_batch == num_batches_per_epoch / 2 - 1:
            # after each half epoch
            epoch += 0.5
            # print "Epoch " + str(epoch)
            H = X_train.dot(W_xh) + b_h
            P = H.dot(W_hp) + b_p
            E = square_loss(P, X_train)
            print E,

        cur_batch = (cur_batch + 1) % X_batches.shape[0]
    print ""
    return W_xh, b_h

def encode_features(X_raw, W, b):
    return X_raw.dot(W) + b

def solve():
    # 300 * 20000
    # Y labels +1/-1
    X_train = load_data("dexter_train.data")
    Y_train = load_labels("dexter_train.labels")
    X_valid = load_data("dexter_valid.data")
    Y_valid = load_labels("dexter_valid.labels")
    X_valid, X_test = X_valid[0:100], X_valid[100:]
    Y_valid, Y_test = Y_valid[0:100], Y_valid[100:]

    print "data loaded"

    clf = SVC(kernel="linear")
    clf.fit(X_train, Y_train)
    prediction = clf.predict(X_test)
    accuracy = 1 - loss01(prediction, Y_test) * 1.0 / Y_test.shape[0]
    print "all data, test=" + str(accuracy)

    dim_range = range(1,10) + range(10,20)[::2] + range(20,50)[::5] + range(50, 210)[::10]

    from keras.layers import Input, Dense
    from keras.models import Model
    from keras.optimizers import SGD
    from keras import regularizers
    from keras.callbacks import EarlyStopping

    # for i in range(10):
    #     plt.plot(X_train[:,i], 'x')
    # plt.show()
    accuracies = []

    for encoding_dim in dim_range:
        # W, b = train_2_layer_autoencoder(X_train, encoding_dim)

        # this is our input placeholder
        input_vec = Input(shape=(X_train.shape[1],))

        # add a Dense layer with a L1 activity regularizer
        encoded = Dense(encoding_dim, activation='relu')(input_vec)
        decoded = Dense(X_train.shape[1], activation='sigmoid')(encoded)
        
        # this model maps an input to its reconstruction
        autoencoder = Model(input=input_vec, output=decoded)
        
        # this model maps an input to its encoded representation
        encoder = Model(input=input_vec, output=encoded)

        # this model maps an input to its encoded representation
        # encoder = Model(input=input_vec, output=encoded)

        sgd = SGD(lr=3.)
        # autoencoder.compile(optimizer=sgd, loss='binary_crossentropy')
        autoencoder.compile(optimizer=sgd, loss='mse')
        # stops training if for patience epochs no validation loss improvement
        early_stopping = EarlyStopping(monitor='val_loss', patience=20)

        autoencoder.fit(X_train, X_train,
                        nb_epoch=100,
                        batch_size=20,
                        shuffle=True,
                        validation_data=(X_valid, X_valid),
                        callbacks=[early_stopping])

        X_train_encoded = encoder.predict(X_train)
        X_valid_encoded = encoder.predict(X_valid)
        X_test_encoded = encoder.predict(X_test)

        # X_train_encoded = encode_features(X_train, W, b)
        # X_valid_encoded = encode_features(X_valid, W, b)

        clf = SVC(kernel="linear")
        clf.fit(X_train_encoded, Y_train)
        prediction = clf.predict(X_test_encoded)
        accuracy = 1 - loss01(prediction, Y_test) * 1.0 / Y_test.shape[0]
        accuracies += [accuracy]
        print "dim=" + str(encoding_dim) + "\ttest=" + str(accuracy)

        print dim_range
        print accuracies

    plt.plot(dim_range, accuracies, '-')
    plt.show()


if __name__ == "__main__":
    solve()

