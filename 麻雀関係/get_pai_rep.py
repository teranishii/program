# -*- coding: utf-8 -*-

import numpy as np

path = "agari_data_2009.txt"

fin = open(path, "r")
line = fin.readline()

all_data = []
c = 0
while line :
    dat = line.split(",")
    all_data.append(dat)
    line = fin.readline()
    c += 1
fin.close()
print(c)

x_data = []
y_data = []

import copy
count = [0] * 34

for dat in all_data:
    
    for j in range(34):
        x = [0] * 34
        y = [int(s) for s in dat]
        if int(dat[j]) > 0:
            x[j] = 1
            y[j] -= 1
            x_data.append(x)
            y_data.append(y)
            count[j] += 1

x_data = np.array([np.array(s) for s in x_data])
y_data = np.array([np.array(s) for s in y_data])

from sklearn.model_selection import train_test_split as split
x_train, x_test, y_train, y_test = split(x_data, y_data, train_size=0.8)

dic_num = 34
hidden_num = 500

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(hidden_num, input_dim = dic_num))
model.add(Dense(dic_num, activation = 'linear'))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.summary()

history = model.fit(x_train, y_train,
                    epochs=10, 
                    batch_size=32,
                    validation_data = (x_test, y_test))

pr = []
pr.append([0] * 34)
for i in range(34):
    tmp = [0] * 34
    tmp[i] = 1
    pr.append(tmp)
pr = np.array([np.array(s) for s in pr])

pre = model.predict(pr)

for i in range(1,35):
    print(pr[i])
    print(pre[i])

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper right')
plt.savefig('pai_distr_expre.eps')
plt.show()

model_json_str = model.to_json()
open("pai_dist_exp.json", 'w').write(model_json_str)
model.save_weights("pai_dist_exq_weights.h5")