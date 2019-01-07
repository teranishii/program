# -*- coding: utf-8 -*-

import numpy as np
import sympy
import matplotlib.pyplot as plt
import pandas as pd
#parameter(次元数，ノット数，x軸の範囲)
n = 4
knot_number = 4
x_left = 0
x_right = 10
#knot_value計算
knot_value=[x_left+(((x_right-x_left)/(knot_number+1))*i) for i in range(1,knot_number+2)]
#データ読み込み
df = pd.read_csv('TrainingDataForAssingment5.csv')
data=df.values.T
#昇順に並び替え
sort_data=sorted(dict(zip(data[1],data[2])).items())
#x,yに分ける
x_data = np.array([w for w, i in sort_data])
y_data = np.array([i for w, i in sort_data])

#spline 関数を求める関数
def spline_function(x_data,y_data,n,knot_number,knot_value,show_func=False,plot=False):
    #LSE計算 (beta計算)
    X = np.zeros((len(x_data), knot_number+n), float)
    for i in range(n):
        X[:,i] = x_data**i
    for i in range(knot_number):
        X[:,n+i] = (((x_data<knot_value[i])*knot_value[i]+(x_data>=knot_value[i])*x_data)-knot_value[i])**(n-1)
    
    theta=np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y_data)
    #Trueならspline関数を表示
    if show_func:
        x=sympy.Symbol("x")
        gx=0
        for i in range(n):
            gx=gx+theta[i]*x**(i)        
        extend=sympy.expand(gx)
        print("cubic spline function among (x|"+str(x_left)+" <= x <= "+str(knot_value[0])+") \n-> f(x) =", extend)
        for i in range(knot_number):
            gx=gx+theta[n+i]*(x-knot_value[i])**(n-1)
            extend=sympy.expand(gx)
            print("cubic spline function among (x|"+str(knot_value[i])+" < x <= "+str(knot_value[i+1])+") \n-> f(x) =", extend)
    #spline関数を保存 plotがTrueならspline関数をグラフに表示
    spline_equation=[0]*(knot_number+1)
    equation="theta[0]"
    for i in range(1,n):
        equation=equation+"+theta["+str(i)+"]*x_value**"+str(i)
    spline_equation[0]=equation
    
    x_value=np.arange(x_left,knot_value[0],0.001)
    y_value=eval(spline_equation[0])
    if plot:
        plt.plot(x_value,y_value)    
    for i in range(knot_number):
        x_value=np.arange(knot_value[i],knot_value[i+1],0.001)
        spline_equation[i+1]=spline_equation[i]+"+theta["+str(n+i)+"]*((x_value-"+str(knot_value[i])+")**"+str(n-1)+")"
        y_value=eval(spline_equation[i+1])
        if plot:
            plt.plot(x_value,y_value)
    
    if plot:
        plt.plot(x_data, y_data, 'b.')
        plt.title("polynomial regression use cubic spline function (knot number is "+str(knot_number)+")")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    return spline_equation,theta

#CV_LOOを計算する関数
def calc_CV_LOO(x_data,y_data,knot_number,knot_value):
    CV_LOO=0
    num=-1
    for i in range(len(x_data[(x_left<= x_data) & (x_data<knot_value[0])])):
        num=num+1
        if i==0:
            fx,theta=spline_function(np.delete(x_data,num),np.delete(y_data,num),n,knot_number,knot_value,show_func=False,plot=True)
        else:
            fx,theta=spline_function(np.delete(x_data,num),np.delete(y_data,num),n,knot_number,knot_value,show_func=False,plot=False)            
        x_value=x_data[num]
        y_predict=eval(fx[0])
        CV_LOO += ((y_data[num] - y_predict) ** 2)
    
    for i in range(knot_number):
        for j in range(len(x_data[(knot_value[i] <= x_data) & (x_data <= knot_value[i+1])])):
            num=num+1
            fx,theta=spline_function(np.delete(x_data,num),np.delete(y_data,num),n,knot_number,knot_value,show_func=False,plot=False)
            x_value=x_data[num]
            y_predict=eval(fx[i+1])
            CV_LOO += ((y_data[num] - y_predict) ** 2)
    CV_LOO=CV_LOO/(num+1)
    
    return CV_LOO

#magic_formulaでCV_LOOを計算
fx,theta=spline_function(x_data,y_data,n,knot_number,knot_value,show_func=True,plot=True)
y_predict=[0]*(knot_number+1)
x_value=x_data[(x_left <= x_data) & (x_data<=knot_value[0])]
y_predict[0]=eval(fx[0])
for i in range(knot_number):
    x_value=x_data[(knot_value[i] < x_data) & (x_data <= knot_value[i+1])]
    y_predict[i+1]=eval(fx[i+1])
y_predict_flat=[element for block in y_predict for element in block]

X = np.zeros((len(x_data), knot_number+n), float)
for i in range(n):
    X[:,i] = x_data**i
for i in range(knot_number):
    X[:,n+i] = (((x_data<knot_value[i])*knot_value[i]+(x_data>=knot_value[i])*x_data)-knot_value[i])**(n-1)
    
h=np.dot(np.dot(X, np.linalg.inv(np.dot(X.T, X))), X.T)
CV_LOO_magic=0
for i in range(len(x_data)):
    CV_LOO_magic += (((y_data[i]-y_predict_flat[i])/(1-h[i][i]))**2)/len(x_data)
print("CV_LOO value use magic formula :" , CV_LOO_magic)
#CV_LOOを計算
CV_LOO=calc_CV_LOO(x_data,y_data,knot_number,knot_value)
print("CV_LOO value not use magic formula :" , CV_LOO)
#ノット数を1~15まで変化させて計算
for i in range(1,16):
    knot_value=[x_left+(((x_right-x_left)/(i+1))*j) for j in range(1,i+2)]
    CV_LOO=calc_CV_LOO(x_data,y_data,i,knot_value)
    print("CV_LOO value not use magic formula ( knot number : "+str(i)+" ):" , CV_LOO)