# -*- coding: utf-8 -*-
import numpy as np
import os

import rouge_calculate

#ROUGE比較テキストのパス(oriが比較元)
ori_text_path="sample_answer.txt"
con_text_path="sample_conpare.txt"

if not os.path.isdir("./ROUGEscore"):
    os.mkdir("./ROUGEscore")

#ROUGEの再現率，適合率，F値の計算(一般的にはROUGEとして再現率Recallが使われる．)
"""
Recall:正解となる文中の単語をどれだけあてられたか
Precision：生成した要約中の単語が，どれだけ正解文の中にあったか
F値：いわゆるF値
"""
Recall=True
Precision=True
Fscore=True

#ROUGEの選択(defaltですべてON)
"""
ROUGE-N : n-gramの 一致数で評価
ROUGE-S : skip-bigram の一致数で評価
ROUGE-SU : skip-bigram + 1-gram の一致数で評価
ROUGE-L : lcs (最長共通部分列)の最大長で評価
"""
R1=True
R2=True
R3=True
RS=True
RSU=True
RL=True

#recallだけ一つのファイルに出力
REonly=True

def main():    
    input_text= open(ori_text_path,encoding="utf-8").read().split('\n')
    conpare_text= open(con_text_path,encoding="utf-8").read().split('\n')
    rouge=rouge_calculate.Rouge(input_text,conpare_text)
    
    if R1:
        rougescore1=rouge.rouge1(Recall,Precision,Fscore)
        rougescore1_ave=np.average(np.array(rougescore1),axis=1)
        
        f=open("ROUGEscore/ROUGE-1.txt","w")
        f.write("ROUGE-1(Recall) :   (average:"+str(rougescore1_ave[0])+")\n"+str(rougescore1[0])+"\n\n")
        f.write("ROUGE-1(Precision) :   (average:"+str(rougescore1_ave[1])+")\n"+str(rougescore1[1])+"\n\n")
        f.write("ROUGE-1(F-score) :   (average:"+str(rougescore1_ave[2])+")\n"+str(rougescore1[2]))
        f.close
    
    if R2:
        rougescore2=rouge.rouge2(Recall,Precision,Fscore)
        rougescore2_ave=np.average(np.array(rougescore2),axis=1)
        
        f=open("ROUGEscore/ROUGE-2.txt","w")
        f.write("ROUGE-2(Recall) :   (average:"+str(rougescore2_ave[0])+")\n"+str(rougescore2[0])+"\n\n")
        f.write("ROUGE-2(Precision) :   (average:"+str(rougescore2_ave[1])+")\n"+str(rougescore2[1])+"\n\n")
        f.write("ROUGE-2(F-score) :   (average:"+str(rougescore2_ave[2])+")\n"+str(rougescore2[2]))
        f.close
    
    if R3:
        rougescore3=rouge.rouge3(Recall,Precision,Fscore)
        rougescore3_ave=np.average(np.array(rougescore3),axis=1)
        
        f=open("ROUGEscore/ROUGE-3.txt","w")
        f.write("ROUGE-3(Recall) :   (average:"+str(rougescore3_ave[0])+")\n"+str(rougescore3[0])+"\n\n")
        f.write("ROUGE-3(Precision) :   (average:"+str(rougescore3_ave[1])+")\n"+str(rougescore3[1])+"\n\n")
        f.write("ROUGE-3(F-score) :   (average:"+str(rougescore3_ave[2])+")\n"+str(rougescore3[2]))
        f.close
    
    if RS:
        rougescores=rouge.rouges(Recall,Precision,Fscore)
        rougescores_ave=np.average(np.array(rougescores),axis=1)
        
        f=open("ROUGEscore/ROUGE-S.txt","w")
        f.write("ROUGE-S(Recall) :   (average:"+str(rougescores_ave[0])+")\n"+str(rougescores[0])+"\n\n")
        f.write("ROUGE-S(Precision) :   (average:"+str(rougescores_ave[1])+")\n"+str(rougescores[1])+"\n\n")
        f.write("ROUGE-S(F-score) :   (average:"+str(rougescores_ave[2])+")\n"+str(rougescores[2]))
        f.close
    
    if RSU:
        rougescoresu=rouge.rougesu(Recall,Precision,Fscore)
        rougescoresu_ave=np.average(np.array(rougescoresu),axis=1)
        
        f=open("ROUGEscore/ROUGE-SU.txt","w")
        f.write("ROUGE-SU(Recall) :   (average:"+str(rougescoresu_ave[0])+")\n"+str(rougescoresu[0])+"\n\n")
        f.write("ROUGE-SU(Precision) :   (average:"+str(rougescoresu_ave[1])+")\n"+str(rougescoresu[1])+"\n\n")
        f.write("ROUGE-SU(F-score) :   (average:"+str(rougescoresu_ave[2])+")\n"+str(rougescoresu[2]))
        f.close
    
    if RL:
        rougescorel=rouge.rougel(Recall,Precision,Fscore)
        rougescorel_ave=np.average(np.array(rougescorel),axis=1)
        
        f=open("ROUGEscore/ROUGE-L.txt","w")
        f.write("ROUGE-L(Recall) :   (average:"+str(rougescorel_ave[0])+")\n"+str(rougescorel[0])+"\n\n")
        f.write("ROUGE-L(Precision) :   (average:"+str(rougescorel_ave[1])+")\n"+str(rougescorel[1])+"\n\n")
        f.write("ROUGE-L(F-score) :   (average:"+str(rougescorel_ave[2])+")\n"+str(rougescorel[2]))
        f.close
    
    if Recall and R1 and R2 and R3 and RS and RSU and RL and REonly:
        f=open("ROUGEscore/ROUGE-Recall.txt","w")
        f.write(str(rougescore1_ave[0])+"\n")
        f.write(str(rougescore2_ave[0])+"\n")
        f.write(str(rougescore3_ave[0])+"\n")
        f.write(str(rougescores_ave[0])+"\n")
        f.write(str(rougescoresu_ave[0])+"\n")
        f.write(str(rougescorel_ave[0])+"\n")
        f.close()

if __name__=="__main__":
    main()