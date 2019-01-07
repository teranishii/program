# -*- coding: utf-8 -*-
import numpy as np

class Rouge():
    def __init__(self,original,conpare):
        word_split_original=[]
        word_split_conpare=[]
        for i in range(len(original)-1):
            word_split = original[i].split()
            word_split_original.append(word_split)
            word_split = conpare[i].split()
            word_split_conpare.append(word_split)
        self.word_split_original=word_split_original
        self.word_split_conpare=word_split_conpare
        self.sentence_len=len(original)-1
    
    
    def rouge1(self,Recall=True,Precision=True,Fscore=True):
        rouge1=[]
        
        if Recall:
            rouge1_re=self.calculate_rouge(self.sentence_len,self.word_split_original,self.word_split_conpare)
        
        if Precision:
            rouge1_pr=self.calculate_rouge(self.sentence_len,self.word_split_conpare,self.word_split_original)
        
        if Recall and Precision and Fscore:
            rouge1_f=self.calculate_f_score(self.sentence_len,rouge1_re,rouge1_pr)
        
        rouge1.append(rouge1_re)
        rouge1.append(rouge1_pr)
        rouge1.append(rouge1_f)
        return rouge1
    
    def rouge2(self,Recall=True,Precision=True,Fscore=True):
        rouge2=[]
        bigram_ori=self.make_bigram(self.sentence_len,self.word_split_original)
        bigram_con=self.make_bigram(self.sentence_len,self.word_split_conpare)
        
        if Recall:
            rouge2_re=self.calculate_rouge(self.sentence_len,bigram_ori,bigram_con)
        
        if Precision:
            rouge2_pr=self.calculate_rouge(self.sentence_len,bigram_con,bigram_ori)
        
        if Recall and Precision and Fscore:
            rouge2_f=self.calculate_f_score(self.sentence_len,rouge2_re,rouge2_pr)
        rouge2.append(rouge2_re)
        rouge2.append(rouge2_pr)
        rouge2.append(rouge2_f)
        return rouge2
    
    def rouge3(self,Recall=True,Precision=True,Fscore=True):
        rouge3=[]
        trigram_ori=self.make_trigram(self.sentence_len,self.word_split_original)
        trigram_con=self.make_trigram(self.sentence_len,self.word_split_conpare)
        
        if Recall:
            rouge3_re=self.calculate_rouge(self.sentence_len,trigram_ori,trigram_con)
        
        if Precision:
            rouge3_pr=self.calculate_rouge(self.sentence_len,trigram_con,trigram_ori)
        
        if Recall and Precision and Fscore:
            rouge3_f=self.calculate_f_score(self.sentence_len,rouge3_re,rouge3_pr)
        
        rouge3.append(rouge3_re)
        rouge3.append(rouge3_pr)
        rouge3.append(rouge3_f)
        return rouge3
    
    def rouges(self,Recall=True,Precision=True,Fscore=True):
        rouges=[]
        skip_bigram_ori=self.make_skipbigram(self.sentence_len,self.word_split_original)
        skip_bigram_con=self.make_skipbigram(self.sentence_len,self.word_split_conpare)
        
        if Recall:
            rouges_re=self.calculate_rouge(self.sentence_len,skip_bigram_ori,skip_bigram_con)
        
        if Precision:
            rouges_pr=self.calculate_rouge(self.sentence_len,skip_bigram_con,skip_bigram_ori)
        
        if Recall and Precision and Fscore:
            rouges_f=self.calculate_f_score(self.sentence_len,rouges_re,rouges_pr)
        
        rouges.append(rouges_re)
        rouges.append(rouges_pr)
        rouges.append(rouges_f)
        return rouges
    
    def rougesu(self,Recall=True,Precision=True,Fscore=True):
        rougesu=[]
        skip_bigram_ori=self.make_skipbigram_in_word(self.sentence_len,self.word_split_original)
        skip_bigram_con=self.make_skipbigram_in_word(self.sentence_len,self.word_split_conpare)
        
        if Recall:
            rougesu_re=self.calculate_rouge(self.sentence_len,skip_bigram_ori,skip_bigram_con)
        
        if Precision:
            rougesu_pr=self.calculate_rouge(self.sentence_len,skip_bigram_con,skip_bigram_ori)
        
        if Recall and Precision and Fscore:
            rougesu_f=self.calculate_f_score(self.sentence_len,rougesu_re,rougesu_pr)
        
        rougesu.append(rougesu_re)
        rougesu.append(rougesu_pr)
        rougesu.append(rougesu_f)
        return rougesu
    
    def rougel(self,Recall=True,Precision=True,Fscore=True):
        rougel=[]
        rougel_re=[]
        rougel_pr=[]
        rougel_f=[]
        
        lcs=self.calculate_lcs(self.sentence_len,self.word_split_original,self.word_split_conpare)
        
        if Recall:
            for i in range(self.sentence_len):
                rougel_re.append(lcs[i]/len(self.word_split_original[i]))
        
        if Recall:
            for i in range(self.sentence_len):
                rougel_pr.append(lcs[i]/len(self.word_split_conpare[i]))
        
        if Recall and Precision and Fscore:
            rougel_f=self.calculate_f_score(self.sentence_len,rougel_re,rougel_pr)
        
        rougel.append(rougel_re)
        rougel.append(rougel_pr)
        rougel.append(rougel_f)
        return rougel
    
    
    def make_bigram(self,length,sentence):
        bigram=[]
        for i in range(length):
            bi_split=[]
            for j in range(len(sentence[i])-1):
                bi_split.append(sentence[i][j]+sentence[i][j+1])
            bigram.append(bi_split)
        return bigram
    
    def make_trigram(self,length,sentence):
        trigram=[]
        for i in range(length):
            tri_split=[]
            for j in range(len(sentence[i])-2):
                tri_split.append(sentence[i][j]+sentence[i][j+1]+sentence[i][j+2])
            trigram.append(tri_split)
        return trigram
    
    def make_skipbigram(self,length,sentence):
        skipbigram=[]
        for i in range(length):
            bi_split=[]
            for j in range(len(sentence[i])):
                for k in range(j+1,len(sentence[i])):
                    bi_split.append(sentence[i][j]+sentence[i][k])
            bi_split=list(set(bi_split))
            skipbigram.append(bi_split)
        return skipbigram
    
    def make_skipbigram_in_word(self,length,sentence):
        skipbigram=[]
        for i in range(length):
            bi_split=[]
            for j in range(len(sentence[i])):
                bi_split.append("<dammy>"+sentence[i][j])
                for k in range(j+1,len(sentence[i])):
                    bi_split.append(sentence[i][j]+sentence[i][k])
            bi_split=list(set(bi_split))
            skipbigram.append(bi_split)
        return skipbigram
    
    def calculate_rouge(self,length,conpared_data,conpare_data):
        SCORE=[]
        for i in range(length):
            count_word=0
            for j in range(len(conpared_data[i])):
                for k in range(len(conpare_data[i])):
                    if conpared_data[i][j]==conpare_data[i][k]:
                        count_word+=1
                        break
            if len(conpared_data[i])==0:
                score=0
            else:
                score=count_word/len(conpared_data[i])
            SCORE.append(score)
        return SCORE
    
    def calculate_f_score(self,length,recall,precision):
        SCORE=[]
        for i in range(length):
            if recall[i]==0 and precision[i]==0:
                F_score=0.0
            else:
                F_score=(2*recall[i]*precision[i])/(recall[i]+precision[i])
            SCORE.append(F_score)
        return SCORE
    
    def calculate_lcs(self,length,split_sen1,split_sen2):
        SCORE=[]
        for i in range(length):
            lcs = [[0 for m in range(len(split_sen2[i]) + 1)] for n in range(len(split_sen1[i])+1)]
            for j in range(1,len(split_sen1[i])+1):
                for k in range(1,len(split_sen2[i])+1):
                    if split_sen1[i][j-1]==split_sen2[i][k-1]:
                        x = 1
                    else:
                        x = 0
                    lcs[j][k]=max(lcs[j-1][k-1]+x,lcs[j-1][k],lcs[j][k-1])
            SCORE.append(lcs[len(split_sen1[i])][len(split_sen2[i])])
        return SCORE


if __name__=="__main__":
    pass