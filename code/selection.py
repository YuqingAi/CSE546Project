from svmutil import *
import numpy as np
from sklearn import decomposition
import fisher

Dim = 20000
lDim = 3
uDim = 20
step = 1
	
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

d = lDim
while d <= uDim:
	X = np.zeros((len(Xtrain), Dim))
	for i in range(len(Xtrain)):
		for j in Xtrain[i]:
			X[i, j] = Xtrain[i][j]
	Y = np.array(Ytrain)
	score = fisher.fisher_score(X, Y)
	rank = fisher.feature_ranking(score)
	XXtrain = X[:, rank[:d]].tolist()

	X = np.zeros((len(Xtest), Dim))
	for i in range(len(Xtest)):
		for j in Xtest[i]:
			X[i, j] = Xtest[i][j]
	XXtest = X[:, rank[:d]].tolist()

	m = svm_train(Ytrain, XXtrain, '-t 0 -q')
	p_label, p_acc, p_val = svm_predict(Ytest, XXtest, m)
	average = p_acc[0]
	
	X = np.zeros((len(Xtest), Dim))
	for i in range(len(Xtest)):
		for j in Xtest[i]:
			X[i, j] = Xtest[i][j]
	Y = np.array(Ytest)
	score = fisher.fisher_score(X, Y)
	rank = fisher.feature_ranking(score)
	XXtest = X[:, rank[:d]].tolist()

	X = np.zeros((len(Xtrain), Dim))
	for i in range(len(Xtrain)):
		for j in Xtrain[i]:
			X[i, j] = Xtrain[i][j]
	XXtrain = X[:, rank[:d]].tolist()

	m = svm_train(Ytest, XXtest, '-t 0 -q')
	p_label, p_acc, p_val = svm_predict(Ytrain, XXtrain, m)
	average = average + p_acc[0]
	average = average / 2
	print(d, average)
	d = d + step