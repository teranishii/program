#coding:utf-8
import math
import numpy as np
from tfidf2 import *

sentences=[]

f=open('b.txt','r')
list=f.readlines()
while '\n' in list:
    list2.remove('\n')
for line in list:
    sentence=line.split()
    sentences.append(sentence)
f.close()

def lex_rank(sentences, n, t,tf,idf):
    cosine_matrix = np.zeros((n, n))
    degrees = np.zeros((n,))
    l = []
    
    for i in range(n):
        for j in range(n):
            cosine_matrix[i][j] = idf_modified_cosine(sentences[i], sentences[j],tf[i],tf[j],idf)
            if cosine_matrix[i][j] > t:
                cosine_matrix[i][j] = 1
                degrees[i] += 1
            else:
                cosine_matrix[i][j] = 0

    for i in range(n):
        for j in range(n):
            cosine_matrix[i][j] = cosine_matrix[i][j] / degrees[i]
    
    ratings = power_method(cosine_matrix, n,10e-6)

    return ratings
    
def power_method(cosine_matrix, n, e):
    transposed_matrix = cosine_matrix.T
    sentences_count = n

    p_vector = np.array([1.0 / sentences_count] * sentences_count)

    lambda_val = 1.0

    while lambda_val > e:
        next_p = np.dot(transposed_matrix, p_vector)
        lambda_val = np.linalg.norm(np.subtract(next_p, p_vector))
        p_vector = next_p

    return p_vector

n=len(sentences)
tf=[0]*n
for i in range(n):
    tf[i]=compute_tf(sentences[i])
idf=compute_idf(sentences)

z=lex_rank(sentences,len(sentences),0.1,tf,idf)
li1, li2 = sorted(z), sorted(range(len(z)), key=lambda k: z[k])
li1.reverse()
li2.reverse()
for i in range(3):
    print(' '.join(sentences[li2[i]]))
