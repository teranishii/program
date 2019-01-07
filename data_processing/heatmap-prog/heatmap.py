# -*- coding: utf-8 -*-

import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def make_bigram(length,sentence):
    bigram=[]
    for i in range(length):
        bi_split=[]
        for j in range(len(sentence[i])-1):
            bi_split.append(sentence[i][j]+sentence[i][j+1])
        bigram.append(bi_split)
    return bigram

HEAT=True
TEXT=True

with open("weight/wei5.txt","r",encoding="utf-8") as f:
    s=f.read().split("<body> ")

a=["","ori_word","gene_word","weight"]

weights=[]
body=s[1].split("<generation> ")[0].replace("\n","").split(" ")

heat=[]

weights.append(body)
weight=s[0].split("\n--------------\n")
number=[i for i in range((len(weight[0].split("\n")))*(len(weight)-1))]

weights=[]
for i in range(len(weight)-1):
    weights+=weight[i].split("\n")

word=["o"+str(i) for i in range(200,200+len(weight[0].split("\n")))]
word=(word)*(len(weight)-1)

genes=["g"+str(i) for i in range(200,200+len(weight)-1)]
genes=(genes)*(len(weight[0].split("\n")))
genes.sort()

heat.append(number)
heat.append(word)
heat.append(genes)
heat.append(weights)
heat=np.array(heat)
a=np.array(a)
heats=np.vstack((a,heat.T))

f = open('heatmap.csv', 'w',newline="",encoding="utf-8")
writer = csv.writer(f)
writer.writerows(heats)
f.close()

generate=s[1].split("<generation> ")[1].replace("\n","").split(" ")
origin=[s[1].split("<generation> ")[0].replace("\n","").split(" ")]
origin_bigram=make_bigram(len(origin),origin)
print(generate)
print(origin)
print(origin_bigram)

if HEAT:
    heatmap = pd.read_csv('heatmap.csv',encoding="utf-8")
    heatmap = heatmap.pivot("ori_word","gene_word","weight")
    sns.set(font=['IPAexGothic'])
    fig, ax = plt.subplots(figsize=(10, 10)) 
    sns.heatmap(heatmap, square=False, vmax=1, vmin=0, center=0.5 , cmap="Blues",linewidths=.5)
    plt.savefig('temp.png')

if TEXT:
    text=[]
    text.append(s[1].split("<generation> ")[0].replace("\n","").split(" "))
    text=np.array(text)
    f = open('heatmap/original.csv', 'w',newline="",encoding="sjis")
    writer = csv.writer(f)
    writer.writerows(text.T)
    f.close()

    text=[]
    text.append(s[1].split("<generation> ")[1].replace("\n","").split(" "))
    text[0].reverse()
    text=np.array(text)
    f = open('heatmap/generation-1gram.csv', 'w',newline="",encoding="sjis")
    writer = csv.writer(f)
    writer.writerows(text)
    f.close()

    generate=s[1].split("<generation> ")[1].replace("\n","").split(" ")
    origin=[s[1].split("<generation> ")[0].replace("\n","").split(" ")]
    origin_bigram=make_bigram(len(origin),origin)
    f = open('heatmap/generation_2gram.csv', 'w',newline="",encoding="sjis")
    writer = csv.writer(f)
    writer.writerows(text.T)
    f.close()