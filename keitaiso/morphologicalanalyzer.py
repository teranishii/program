# -*- coding: utf-8 -*-

import subprocess as proc
import sys
import io
import os
import time


class Mecab():

    def __init__(self):
        self.name   = "Mecab"
        self.char   = "utf-8"
        self.b_size = 8192 * 3
        self.s_size = 0
        self.temp   = "bbb.txt"



    ## 形態素解析して表層表現の集合を得る.
    #INPUT# data(String): 形態素解析したい文もしくは文が入ったファイルのパス
    #INPUT# is_file(Boolean): data がファイルパスであるかどうか
    #RETURN# results(List_String): 形態素解析後の表層表現の配列
    def get_surface(self, data, is_file=False):
        self.s_size = 0
        if is_file:
            sentences = []
            with open(data, "r", encoding=self.char) as f:
                for sentence in f:
                    sentences.append(sentence)
                    self.s_size += 1
            if "\ufeff" in sentences[0]:
                sentences[0] = sentences[0].replace("\ufeff", "")
            with open(data, "w", encoding=self.char) as f:
                for sentence in sentences:
                    f.write(sentence)
            file_name = data
        else:
            with open(self.temp, "w", encoding=self.char) as f:
                f.write(data)
                self.s_size += 1
            file_name = self.temp
        cmd = ["mecab"]
        cmd.append("-F%m\s")
        cmd.append("-U%m\s")
        cmd.append("-E\n")
        cmd.append("-b")
        cmd.append(str(self.b_size))
        cmd.append(file_name)
        (outputs, error) = proc.Popen(cmd, stdout=proc.PIPE).communicate()
        results = outputs.decode(self.char).split(" \n")[:-1]
        return results



    ## 形態素解析して見出し語の集合を得る.
    #INPUT# data(String): 形態素解析したい文もしくは文が入ったファイルのパス
    #INPUT# is_file(Boolean): data がファイルパスであるかどうか
    #RETURN# results(List_String): 形態素解析後の見出し語の配列
    def get_lemma(self, data, is_file=False):
        self.s_size = 0
        if is_file:
            sentences = []
            with open(data, "r", encoding=self.char) as f:
                for sentence in f:
                    sentences.append(sentence)
                    self.s_size += 1
            if "\ufeff" in sentences[0]:
                sentences[0] = sentences[0].replace("\ufeff", "")
            with open(data, "w", encoding=self.char) as f:
                for sentence in sentences:
                    f.write(sentence)
            file_name = data
        else:
            with open(self.temp, "w", encoding=self.char) as f:
                f.write(data)
            file_name = self.temp
        cmd = ["mecab"]
        cmd.append("-F%f[6]\s")
        cmd.append("-U%m\s")
        cmd.append("-E\n")
        cmd.append("-b")
        cmd.append(str(self.b_size))
        cmd.append(file_name)
        (outputs, error) = proc.Popen(cmd, stdout=proc.PIPE).communicate()
        results = outputs.decode(self.char).split(" \n")[:-1]
        return results



    ## 形態素解析して品詞の集合を得る.
    #INPUT# data(String): 形態素解析したい文もしくは文が入ったファイルのパス
    #INPUT# is_file(Boolean): data がファイルパスであるかどうか
    #INPUT# is_deep(Boolean): 詳細な品詞情報を取得するかどうか
    #RETURN# results(List_String): 形態素解析後の品詞の配列
    def get_pos(self, data, is_file=False, is_deep=False):
        self.s_size = 0
        if is_file:
            sentences = []
            with open(data, "r", encoding=self.char) as f:
                for sentence in f:
                    sentences.append(sentence)
                    self.s_size += 1
            if "\ufeff" in sentences[0]:
                sentences[0] = sentences[0].replace("\ufeff", "")
            with open(data, "w", encoding=self.char) as f:
                for sentence in sentences:
                    f.write(sentence)
            file_name = data
        else:
            with open(self.temp, "w", encoding=self.char) as f:
                f.write(data)
                self.s_size += 1
            file_name = self.temp
        cmd = ["mecab"]
        if is_deep:
            cmd.append("-F%H\s\n")
            cmd.append("-Uunk,unk,unk,*,*,*,*,*,*\s\n")
            cmd.append("-EEOS\s\n")
        else:
            cmd.append("-F%f[0]\s")
            cmd.append("-Uunk\s")
            cmd.append("-E\n")
        cmd.append("-b")
        cmd.append(str(self.b_size))
        cmd.append(file_name)
        (outputs, error) = proc.Popen(cmd, stdout=proc.PIPE).communicate()
        outputs = outputs.decode(self.char).split(" \n")[:-1]
        if is_deep:
            results = []
            result  = []
            for output in outputs:
                output = output.rstrip()
                if output != "EOS":
                    result.append("-" + "-".join(output.split(",")[0:6]) + "-")
                else:
                    results.append(" ".join(result))
                    result = []
        else:
            results = outputs
        return results



    ## 形態素解析して品詞番号の集合を得る.
    #INPUT# data(String): 形態素解析したい文もしくは文が入ったファイルのパス
    #INPUT# is_file(Boolean): data がファイルパスであるかどうか
    #RETURN# results(List_String): 形態素解析後の品詞番号の配列
    def get_pos_number(self, data, is_file=False):
        self.s_size = 0
        if is_file:
            sentences = []
            with open(data, "r", encoding=self.char) as f:
                for sentence in f:
                    sentences.append(sentence)
                    self.s_size += 1
            if "\ufeff" in sentences[0]:
                sentences[0] = sentences[0].replace("\ufeff", "")
            with open(data, "w", encoding=self.char) as f:
                for sentence in sentences:
                    f.write(sentence)
            file_name = data
        else:
            with open(self.temp, "w", encoding=self.char) as f:
                f.write(data)
                self.s_size += 1
            file_name = self.temp
        cmd = ["mecab"]
        cmd.append("-F%h\s")
        cmd.append("-U69\s")
        cmd.append("-E\n")
        cmd.append("-b")
        cmd.append(str(self.b_size))
        cmd.append(file_name)
        (outputs, error) = proc.Popen(cmd, stdout=proc.PIPE).communicate()
        results = outputs.decode(self.char).split(" \n")[:-1]
        return results



    ## 形態素解析した文の文章数を返す.
    #INPUT# なし
    #RETURN# self.s_size(Integer): 形態素解析した文の文章数
    def get_sentence_size():
        return self.s_size






## テスト用のメイン関数
if __name__ == "__main__":
    mecab = Mecab()

    s = "aaaa"
    f = "bbb.txt"

    start = time.time()
    lemma   = mecab.get_lemma(s)
    surface = mecab.get_surface(s)
    pos     = mecab.get_pos(s)
    pos_d   = mecab.get_pos(s, is_deep=True)
    pos_num = mecab.get_pos_number(s)
    print(s)
    print("lemma:", lemma)
    print("surface:", surface)
    print("pos:", pos)
    print("pos_d:", pos_d)
    print("pos_num:", pos_num)
    print()
    """
    lemma   = mecab.get_lemma(f, is_file=True)
    surface = mecab.get_surface(f, is_file=True)
    pos     = mecab.get_pos(f, is_file=True)
    pos_d   = mecab.get_pos(f, is_file=True, is_deep=True)
    pos_num = mecab.get_pos_number(f, is_file=True)
    print("lemma:", lemma)
    print("surface:", surface)
    print("pos:", pos)
    print("pos_d:", pos_d)
    print("pos_num:", pos_num)

    end = time.time()
    print("Time:", end-start)
    """
