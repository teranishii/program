# -*- coding: utf-8 -*-

from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.grid_search import GridSearchCV
import re
import numpy as np
import preprocess
import os

# 学習
en_path = os.path.join("./", "train/body.txt")
source_vocab = ['<eos>', '<unk>', '<bos>'] + preprocess.count_words(en_path, 1000)
source_data = preprocess.make_dataset(en_path, source_vocab)
source_ids = {word: index for index, word in enumerate(source_vocab)}
words = {i: w for w, i in source_ids.items()}
N = len(source_data)

words[len(words)]="padding"

a=[0]*(1000)
b=[1]*(1000)

max_len=0
for k in range(len(source_data)):
    if max_len<len(source_data[k]):
        max_len=len(source_data[k])
for k in range(len(source_data)):
    source_data[k]=source_data[k][:max_len] + [-1]*(max_len-len(source_data[k]))

a_on = np.identity(len(words))[source_data]
a_on=np.array(a_on,dtype="float32")
a_on=a_on[:,:,0:len(words)-1]
a_on=np.sum(a_on,axis=1)

train_data=np.concatenate((a_on[0:900],a_on[1000:1900]),axis=0)
test_data=np.concatenate((a_on[900:1000],a_on[1900:2000]),axis=0)
train_label=a[0:900]+b[0:900]
test_label=a[900:1000]+b[900:1000]

clf = svm.SVC(kernel='rbf', C=1, gamma=0.1)
clf.fit(train_data,train_label)
test_pred = clf.predict(test_data)
print(accuracy_score(test_label, test_pred))

clf = svm.SVC(kernel='rbf', C=1, gamma=0.01)
clf.fit(train_data,train_label)
test_pred = clf.predict(test_data)
print(accuracy_score(test_label, test_pred))

clf = svm.SVC(kernel='rbf', C=1, gamma=0.001)
clf.fit(train_data,train_label)
test_pred = clf.predict(train_data)
print(accuracy_score(train_label, test_pred))


clf = svm.SVC(kernel='rbf', C=10, gamma=0.1)
clf.fit(train_data,train_label)
test_pred = clf.predict(test_data)
print(accuracy_score(test_label, test_pred))
clf = svm.SVC(kernel='rbf', C=10, gamma=0.01)
clf.fit(train_data,train_label)
test_pred = clf.predict(test_data)
print(accuracy_score(test_label, test_pred))

clf = svm.SVC(kernel='rbf', C=10, gamma=0.001)
clf.fit(train_data,train_label)
test_pred = clf.predict(test_data)
print(accuracy_score(test_label, test_pred))


clf = svm.SVC(kernel='rbf', C=100, gamma=0.1)
clf.fit(train_data,train_label)
test_pred = clf.predict(test_data)
print(accuracy_score(test_label, test_pred))
clf = svm.SVC(kernel='rbf', C=100, gamma=0.01)
clf.fit(train_data,train_label)
test_pred = clf.predict(test_data)
print(accuracy_score(test_label, test_pred))

clf = svm.SVC(kernel='rbf', C=100, gamma=0.001)
clf.fit(train_data,train_label)
test_pred = clf.predict(test_data)
print(accuracy_score(test_label, test_pred))
