# -*- coding: utf-8 -*-

import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob

def make_bigram(length,sentence):
    bigram=[]
    for i in range(length):
        bi_split=[]
        for j in range(len(sentence[i])-1):
            bi_split.append(sentence[i][j]+sentence[i][j+1])
        bigram.append(bi_split)
    return bigram

with open("weight/wei.txt","r",encoding="utf-8") as f:
    s=f.read().split("<body> ")
body=s[1].split("<generation> ")[1].replace("\n","").split(" ")
length=len(s[1].split("<generation> ")[0].replace("\n","").split(" "))

weight_ave=[]

INPUT_PATH = "weight/*.txt"
filepath = glob.glob(INPUT_PATH)
for path in filepath:
    with open(path,encoding="utf-8") as f:
        sentence=f.read().split("<body> ")
        value=sentence[0].split("\n--------------\n")
        weight=[]
        for i in range(len(value)-1):
            values=value[i].split("\n")
            value2=[float(j) for j in values]
            assert(len(value2)==length)
            weight.append(value2)
    weight_ave.append(weight)

weight_ave=np.array(weight_ave)

average=np.average(weight_ave,axis=0)
variance=np.var(weight_ave,axis=0)

body=s[1].split("<generation> ")[0].replace("\n","")
generation=s[1].split("<generation> ")[1].replace("\n","")

with open("weight/calculate/average.txt","w",encoding="utf-8") as f:
    for i in range(len(average)):
        for j in range(len(average[0])):
            f.write(str(average[i][j])+"\n")
        f.write("--------------\n")
    f.write("<body> "+str(body)+"\n")
    f.write("<generation> "+str(generation)+"\n")

with open("weight/calculate/variance.txt","w",encoding="utf-8") as f:
    for i in range(len(variance)):
        for j in range(len(variance[0])):
            f.write(str(variance[i][j])+"\n")
        f.write("--------------\n")
    f.write("<body> "+str(body)+"\n")
    f.write("<generation> "+str(generation)+"\n")