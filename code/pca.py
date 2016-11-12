from svmutil import *
import numpy as np
from sklearn import decomposition

Dim1 = 20000
Dim2 = 80
train_data_file = '../data/dexter/train_data.txt'
train_label_file = '../data/dexter/train_label.txt'
test_data_file = '../data/dexter/val_data.txt'
test_label_file = '../data/dexter/val_label.txt'
Xtrain = []
Ytrain = []
Xtest = []
Ytest = []

with open(train_data_file) as f:
    content = f.readlines()
for lines in content:
	entries = lines.split(' ')
	features = dict()
	for entry in entries:
		tmp = entry.split(':')
		if len(tmp) == 2:
			features[int(tmp[0])] = int(tmp[1])
	Xtrain.append(features)

with open(train_label_file) as f:
    content = f.readlines()
for lines in content:
	Ytrain.append(int(lines))

with open(test_data_file) as f:
    content = f.readlines()
for lines in content:
	entries = lines.split(' ')
	features = dict()
	for entry in entries:
		tmp = entry.split(':')
		if len(tmp) == 2:
			features[int(tmp[0])] = int(tmp[1])
	Xtest.append(features)

with open(test_label_file) as f:
    content = f.readlines()
for lines in content:
	Ytest.append(int(lines))
"""
R = np.random.normal(0, 1, (Dim1, Dim2))

X = np.zeros((len(Xtrain), Dim1))
for i in range(len(Xtrain)):
	for j in Xtrain[i]:
		X[i, j] = Xtrain[i][j]
X = X.dot(R)
Xtrain = X.tolist()

X = np.zeros((len(Xtest), Dim1))
for i in range(len(Xtest)):
	for j in Xtest[i]:
		X[i, j] = Xtest[i][j]
X = X.dot(R)
Xtest = X.tolist()
print(Xtest)
"""
X = np.zeros((len(Xtrain), Dim1))
for i in range(len(Xtrain)):
	for j in Xtrain[i]:
		X[i, j] = Xtrain[i][j]
pca = decomposition.PCA(n_components = Dim2)
pca.fit(X)
Xtrain = pca.transform(X).tolist()
X = np.zeros((len(Xtest), Dim1))
for i in range(len(Xtest)):
	for j in Xtest[i]:
		X[i, j] = Xtest[i][j]
Xtest = pca.transform(X).tolist()


m = svm_train(Ytrain, Xtrain, '-c 0.1 -t 0')
p_label, p_acc, p_val = svm_predict(Ytest, Xtest, m)
print(p_label)