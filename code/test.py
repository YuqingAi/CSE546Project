from svmutil import *
import numpy as np

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

m = svm_train(Ytrain, Xtrain, '-c 0.0001')
p_label, p_acc, p_val = svm_predict(Ytest, Xtest, m)
print(p_label)