import numpy as np
import scipy.sparse as sp
# import matplotlib.pyplot as plt
import math

def get_initial_lambda(X, Y):
	temp = Y - np.average(Y)
	temp = X.T.dot(temp)
	return np.amax(np.absolute(temp)) * 2

def precompute_a(X):
	N, d = X.shape
	a = [0] * d
	for k in range(d):
		col = X[:,k]
		a[k] = col.T.dot(col)[0, 0] * 2
	return np.array(a).T

def solve_lasso_with_lamb(X, Y, lamb, w = None, w_0 = None):
	N, d = X.shape

	if w is None:
		w = np.ones(d).T
	if w_0 is None:
		w_0 = 0

	t = 0
	delta_w = float("inf")
	stop_cond = 0.1

	a = precompute_a(X)  # a does not change, dx1
	
	while delta_w > stop_cond:
		y_hat = X.dot(w) + w_0  # y_hat recalculation
		w_0_new = np.sum(Y - y_hat) / N + w_0
		w_new = w.copy()
		y_hat_new = y_hat - w_0 + w_0_new 

		for k in range(d):
			c_k = 2 * X[:, k].T.dot(Y - y_hat_new)[0] + a[k] * w_new[k]
			
			if (c_k < -lamb):
				w_new[k] = (c_k + lamb) / a[k]
			elif (c_k > lamb):
				w_new[k] = (c_k - lamb) / a[k]
			else:
				w_new[k] = 0

			X_k_col = np.squeeze(sp.spmatrix.toarray(X[:,k]))
			y_hat_new = y_hat_new - np.multiply(X_k_col, w[k] - w_new[k])

		if t > 0:
			delta_w = np.amax(np.absolute(w_new - w))
		# print t, lasso_loss(X, Y, w_new, w_0_new, lamb)
		t += 1
		w = w_new
		w_0 = w_0_new

	return w, w_0

def lasso_loss(X, Y, w, w_0, lamb):
	temp = X.dot(w) - (Y - w_0)
	loss = temp.T.dot(temp)
	reg = np.sum(np.absolute(w)) * lamb
	return loss + reg

def sse(X, Y, w, w_0):
	temp = X.dot(w) - (Y - w_0)
	return temp.T.dot(temp)

def rmse(X, Y, w, w_0):
	N, d = X.shape
	return math.sqrt(sse(X, Y, w, w_0) / N)

def num_nonzeros(w):
	res = 0
	for i in range(len(w)):
		if (abs(w[i]) > 1e-09):
			res += 1
	return res

def solve_lasso(X_train, Y_train, X_valid, Y_valid):
	lamb = 10
	dec_frac = 0.5
	num_itrs = 20

	w_hat = None
	w_0_hat = None

	w_hat_2 = None
	w_0_hat_2 = None
	best_lamb = None
	best_loss = float("inf")

	for i in range(num_itrs):
		# uses previous sol as initial conditions for efficiency
		w_hat, w_0_hat = solve_lasso_with_lamb(X_train, Y_train, lamb, w_hat, w_0_hat)
		w_hat_2, w_0_hat_2 = solve_lasso_with_lamb(X_valid, Y_valid, lamb, w_hat_2, w_0_hat_2)
		
		rmse1 = rmse(X_valid, Y_valid, w_hat, w_0_hat)
		rmse2 = rmse(X_train, Y_train, w_hat_2, w_0_hat_2)
		num_nz = num_nonzeros(w_hat)

		if (rmse1 + rmse2) / 2.0 < best_loss:
			best_loss = (rmse1 + rmse2) / 2.0
			best_lamb = lamb

		print "lambda=" + str(lamb) + "\trmse=" + str((rmse1 + rmse2) / 2) + "\tnonzero=" + str(num_nz)
		lamb *= dec_frac

	return w_hat, w_0_hat, best_lamb
