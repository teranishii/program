# -*- coding: utf-8 -*-

import glob
import re
import os

INPUT_PATH = "Tweet/train.txt"
filepath = glob.glob(INPUT_PATH)    
for path in filepath:
    lines = open(path,encoding="utf-8").read()
    lines=re.split("\n",lines)
    a=len(lines)-1
    a=a/2
    a=int(a)
    f=open("Tweet/aaa.txt","w",encoding="utf-8")
    for i in range(a):
        f.write("0\t"+str(lines[i])+"\n")
    for i in range(a,2*a):
        f.write("1\t"+str(lines[i])+"\n")
    f.close()