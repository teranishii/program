#!/usr/bin/python
# -*- coding: utf-8 -*-

#218014042 寺西弘樹

def prime(x):
    prime_list=[]
    if x==1:
        print("1 is not prime number.")
    elif 1<x:
        for i in range(2,x+1):
            flag=0
            for j in range(2,i):
                if i%j==0:
                    flag=1
                    break
            if flag==0:
                prime_list.append(i)
    else:
        print("Type positive integer.")
    
    print(prime_list)
    return

if __name__ == '__main__':
    prime(100)