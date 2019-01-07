# -*- coding: utf-8 -*-

from numpy.random import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def rotate_x(deg):
    # degreeをradianに変換
    r = np.radians(deg)
    C = np.cos(r)
    S = np.sin(r)
    # x軸周りの回転行列
    R_x = np.array([[1, 0, 0],
                    [0, C, -S],
                    [0, S, C]])

    return R_x
"""
b=randn(5000,3)

d=[]
for i in range(5000):
    temp=np.array(b[i])
    rx=rotate_x(90)
    v=np.dot(rx,temp)
    v[0]=v[0]*30
    v[1]=v[1]*150
    v[2]=v[2]*200
    d.append(v.tolist())
d=np.array(d)
e=d.T

fig = plt.figure()
ax = Axes3D(fig)


# 軸ラベルの設定
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")

ax.plot(e[0], e[1], e[2], "+", color="green", ms=4, mew=0.5,label="input_data")
ax.legend(numpoints = 1)
plt.show()

"""

import numpy as np

input_data = []
"""
f = open("sampleData.dat","r")
for line in f:
    x = line.split(" ")
    del(x[-1])#改行文字の削除
    a=[float(item) for item in x]
    #temp = [[temp[0]],[temp[1]],[temp[2]]]
    input_data.append(a)    
f.close()
input_data = np.array(input_data[5000:])#1*3の行列
"""


"""
input_data=d

cov = np.cov(input_data,rowvar=0,bias=1)

la,v = np.linalg.eig(cov)
print(la)
print(v)

fig = plt.figure()
ax = Axes3D(fig)

data= np.random.multivariate_normal([0,0,0],[[30,0,0],[0,150,0],[0,0,200]], 5000)
e=data.T
ax.set_xlim(-40, 40)
ax.set_ylim(-40, 40)
ax.set_zlim(-40, 40)

# 軸ラベルの設定
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")

ax.plot(e[0], e[1], e[2], "+", color="green", ms=4, mew=0.5,label="input_data")
ax.legend(numpoints = 1)
plt.show()

sleep()

"""


from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from scipy import genfromtxt

fig = pyplot.figure()
ax = Axes3D(fig)

# 軸ラベルの設定
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")

X = []
Y = []
Z = []
f = open("sampleData.dat","r")

for line in f:
    x = line.split(" ")
    del(x[-1])

    a=[float(item) for item in x]
    X.append(a[0])
    Y.append(a[1])
    Z.append(a[2])
f.close()
del(X[5000:])
del(Y[5000:])
del(Z[5000:])
#print Z
#sys.exit()
# グラフ描画
ax.plot(X, Y, Z, "+", color="green", ms=4, mew=0.5,label="input_data")
ax.legend(numpoints = 1)
#ax.plot(d2[:,0], d2[:,1], d2[:,2], "o", color="#00cccc", ms=4, mew=0.5)
#ax.plot(d3[:,0], d3[:,1], d3[:,2], "o", color="#ff0000", ms=4, mew=0.5)
pyplot.show()

import random
import sys

input_data = []

eta = 0.0001
#データ読み込み

f = open("sampleData.dat","r")
for line in f:
    x = line.split(" ")
    del(x[-1])#改行文字の削除
    a=[float(item) for item in x]
    #temp = [[temp[0]],[temp[1]],[temp[2]]]
    input_data.append(a)    
f.close()

cov = np.cov(input_data,rowvar=0,bias=1)
print(cov)

la,v = np.linalg.eig(cov)

num=np.argsort(la)[::-1]
v=v[:,[num[0],num[1],num[2]]]
la=la[[num[0],num[1],num[2]]]

print(la)
print(v)
weight=v.T[0:2].T
weight = np.matrix(weight)

inv_w = np.linalg.inv(weight.T*weight)
P=weight*inv_w*weight.T
print(P)

input_data = np.matrix(input_data[5000:])#1*3の行列

#重みの用意
weight = [[0 for i in range(2)] for j in range(3)]#weight初期化

for i in range(3):
    for j in range(2):
        weight[i][j] = random.uniform(-1,1)
        #print weight[i][j]

weight = np.matrix(weight)
FirstWeight = weight
#print input_data[0]
#print eta*input_data[0]
#sys.exit()
#重みの更新

print(weight)

for epoch in range(100):#千回更新
    sum1=0
    for i in input_data:#入力データの分だけ
        xxt = i.T*i
        y = weight.T*i.T
        yyt = y*(y.T)
        xxtW = xxt*weight
        Wyyt = weight*yyt
        weight = weight + eta*(xxtW - Wyyt)
        
        inv_w = np.linalg.inv(weight.T*weight)
        P=weight*inv_w*weight.T
        LSE=i.T-P*i.T
        sum1+=sum(map(lambda x: x**2, LSE))
    #print(sum1)
    #print(weight)
print("FirstWeight\n",FirstWeight)
print("new weight\n",weight)

s1=0
s2=0

s1=weight.T[0:1]*v.T[2:3].T
s2=weight.T[1:2]*v.T[2:3].T
print(s1)
print(s2)


sum1=0
sum2=0
sum1+=sum(map(lambda x: x**2, weight.T[0:1].T))
sum2+=sum(map(lambda x: x**2, v.T[0:1].T))

print(weight.T[0:1]*v.T[0:1].T/(np.sqrt(sum1)*np.sqrt(sum2)))