# -*- coding: utf-8 -*-

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
#N:生成する乱数の総数
#thres:閾値
N=100000
thres=2.5
#標準正規分布の Pr(X>2.5) における確率の真値を積分で計算
ans = sp.integrate.quad(sp.stats.norm.pdf, thres, np.inf)[0]

#標準正規分布におけるモンテカルロ近似
#標準正規分布に従う乱数 X を N 個生成
x1=sp.stats.norm.rvs(0,1,size=N)
#確率変数 N 個の標準正規分布ヒストグラムの表示
plt.hist(x1, bins=100)
plt.show()  
#x > 2.5 となる X のみを True としてカウント
x1 = [True if i > 2.5 else False for i in x1]
#True の数を累積カウント
prob=np.cumsum(x1)
#i (0 < i < N+1) 回目の試行におけるモンテカルロ近似の確率を計算し，最後に確率を表示
prob=[prob[i]/i for i in range(1,N)]
montprob=prob[len(prob)-1]
print(montprob)
# N 回試行におけるモンテカルロ近似の確率の推移の表示
plt.plot(prob)
plt.title("The mean of Pr(X > 2.5) for X ~ N(0,1)")
plt.xlabel("n")
plt.ylabel("Probability")
plt.show()

#重点サンプリングを用いたモンテカルロ近似
#平均 3,分散 1 の正規分布に従う X を N 個生成
x2=sp.stats.norm.rvs(3,1,size=N)
#標準正規分布における確率密度関数 f(x) の計算と
#平均 3,分散 1 の正規分布における確率密度関数 g(x) の計算
f_y=sp.stats.norm(0,1).pdf(x2)
g_y=sp.stats.norm(3,1).pdf(x2)
#h~(x) = h(x)*f(x)/g(x)の計算
#ここでh(x) : I(2.5 < x) f(x) : X ~ N(0,1) g(x) : X ~ N(3,1)
prob=[True if i > 2.5 else False for i in x2]*f_y/g_y
#h~(x)を累積カウント
prob=np.cumsum(prob)
#i (0 < i < N+1) 回目の試行における重点サンプリングを用いたモンテカルロ近似の確率を計算し，最後に確率を表示
prob=[prob[i]/i for i in range(1,N)]
montprob_sampling=prob[len(prob)-1]
print(montprob_sampling)
# N 回試行におけるモンテカルロ近似の確率の推移の表示
plt.plot(prob)
plt.title("The mean of Pr(X > 2.5) for X ~ N(0,1) use importance distribution N(3,1)")
plt.xlabel("n")
plt.ylabel("Probability")
plt.show()

print("Collect answer : ", ans)
print("Monte carlo approximation of N(0,1) : ", montprob)
print("Monte carlo approximation of N(0,1) use importance distribution N(3,1) : ", montprob_sampling)