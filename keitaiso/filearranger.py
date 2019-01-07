# -*- coding: utf-8 -*-

import os
import sys
import time
import glob
import numpy as np
import Utils.morphologicalanalyzer as morph
import Utils.syntaxanalyzer as syntax

class FileArranger():
    def __init__(self):
        self.name = "FileArranger"


    ## ファイルを形態素単位に分かち書きする
    #INPUT# i_file_name(String): 分かち書きするファイルのパス
    #INPUT# o_file_name(String): 分かち書きした後のファイルのパス
    #INPUT# i_enc(String): ファイル読み込み時の文字コード
    #INPUT# o_enc(String): ファイル書き込み時の文字コード
    #INPUT# is_lemma(Boolean): 基本形で分かち書きするかどうか
    #RETURN# なし
    def wakati_file(self, i_file_name, o_file_name, i_enc="utf-8", o_enc="utf-8", is_lemma=False):
        mecab = morph.Mecab()
        start = time.time()
        if is_lemma:
            sentences = mecab.get_lemma(i_file_name, is_file=True)
        else:
            sentences = mecab.get_surface(i_file_name, is_file=True)
        with open(o_file_name, "w", encoding=o_enc) as output:
            for sentence in sentences:
                output.write(sentence+"\n")
        end = time.time()
        print("Wakati:", i_file_name, "-->", o_file_name, "\t(time:", (end-start), ")")



    ## ファイルを文節が分かるように形態素単位で分かち書きする.
    #INPUT# i_file_name(String): 分かち書きするファイルのパス
    #INPUT# o_file_name(String): 分かち書きした後のファイルのパス
    #INPUT# i_enc(String): ファイル読み込み時の文字コード
    #INPUT# o_enc(String): ファイル書き込み時の文字コード
    #INPUT# is_lemma(Boolean): 基本形で分かち書きするかどうか
    #RETURN# なし
    def segment_file(self, i_file_name, o_file_name, i_enc="utf-8", o_enc="utf-8", is_lemma=False):
        cabocha = syntax.CaboCha()
        start   = time.time()
        if is_lemma:
            sentences = cabocha.get_segment_lemma(i_file_name, is_file=True)
        else:
            sentences = cabocha.get_segment_surface(i_file_name, is_file=True)
        with open(output_file_name, "w", encoding=o_enc) as output:
            for sentence in sentences:
                output.write(sentence+"\n")
        end = time.time()
        print("Wakati:", i_file_name, "-->", o_file_name, "\t(time:", (end-start), ")")



    ## ファイルを条件付きで形態素単位で分かち書きする.
    ## 名詞, 動詞, 形容詞のグループおよびそれ以外のグループのどちらかを品詞で一括にして分かち書きする.
    #INPUT# i_file_name(String): 分かち書きするファイルのパス
    #INPUT# o_file_name(String): 分かち書きした後のファイルのパス
    #INPUT# i_enc(String): ファイル読み込み時の文字コード
    #INPUT# o_enc(String): ファイル書き込み時の文字コード
    #INPUT# is_lemma(Boolean): 基本形で分かち書きするかどうか
    #INPUT# use_independent(Boolean): 名詞, 動詞, 形容詞以外のグループを品詞で一括化するか
    #RETURN# なし
    def wakati_conditional_file(self, i_file_name, o_file_name, i_enc="utf-8", o_enc="utf-8", is_lemma=False, use_independent=True):
        mecab = morph.Mecab()
        start = time.time()
        if is_lemma:
            sentences1 = mecab.get_lemma(i_file_name, is_file=True)
        else:
            sentences1 = mecab.get_surface(i_file_name, is_file=True)
        sentences2 = mecab.get_pos(i_file_name, is_file=True, is_deep=True)
        with open(o_file_name, "w", encoding=o_enc) as output:
            for sentence1, sentence2 in zip(sentences1, sentences2):
                sentence = []
                words1 = sentence1.split(" ")
                words2 = sentence2.split(" ")
                for word1, word2 in zip(words1, words2):
                    if use_independent:
                        word_a = word1
                        word_b = word2
                    else:
                        word_a = word2
                        word_b = word1
                    if ("-名詞-" in word2) or ("-動詞-" in word2) or ("-形容詞-" in word2):
                        sentence.append(word_a)
                    else:
                        sentence.append(word_b)
                output.write(" ".join(sentence)+"\n")
        end = time.time()
        print("Conditional wakati:", i_file_name, "-->", o_file_name, "\t(time:", (end-start), ")")



    ## 複数のファイルを 1 つのファイルに結合する.
    #INPUT# i_file_names(List_String): 結合したいファイルのパス集合
    #INPUT# o_file_name(String): 結合した後のファイルのパス
    #INPUT# i_enc(String): ファイル読み込み時の文字コード
    #INPUT# o_enc(String): ファイル書き込み時の文字コード
    #RETURN# なし
    def merge_file(self, i_file_names, o_file_name, i_enc="utf-8", o_enc="utf-8"):
        start = time.time()
        with open(o_file_name, "w", encoding=o_enc) as output:
            all = 0
            for file_name in i_file_names:
                count = 0;
                with open(file_name, "r", encoding=i_enc) as f:
                    for line in f:
                        output.write(str(line))
                        count += 1
                print(file_name, ":", count)
                all += count
##            print("merge", str(all), "sentences")
        end = time.time()
        print("Merge:", all, "sentences -->", o_file_name, "\t(time:", (end-start), ")")



    ## 複数のファイルから一定の割合だけランダムに選択して 1 つのファイルに結合する.
    #INPUT# i_file_names(List_String): 結合したいファイルのパス集合
    #INPUT# o_file_name(String): 結合した後のファイルのパス
    #INPUT# i_enc(String): ファイル読み込み時の文字コード
    #INPUT# o_enc(String): ファイル書き込み時の文字コード
    #INPUT# ratio(Float): ファイルからランダムに選択する割合
    #RETURN# なし
    def merge_conditional_file(self, i_file_names, o_file_name, i_enc="utf-8", o_enc="utf-8", ratio=0.5):
        start = time.time()
        with open(o_file_name, "w", encoding=o_enc) as output:
            all = 0
            for file_name in i_file_names:
                count = 0
                with open(file_name, "r", encoding=i_enc) as f:
                    lines = list(f)
                    order = np.random.permutation(int(len(lines)*ratio+0.5))
                    for position in order:
                        output.write(lines[position])
                        count += 1
                print(file_name, ":", count)
                all += count
##            print("merge", str(all), "sentences")
        end = time.time()
        print("Merge:", all, "sentences -->", output_file_name, "\t(time:", (end-start), ")")



    ## ファイルの内容を読み込む.
    #INPUT# i_file_name(String): 読み込みたいファイルのパス
    #INPUT# i_enc(String): ファイル読み込み時の文字コード
    #RETURN# sentences(List_String): 読み込んだファイル内容
    def read_file(self, i_file_name, i_enc="utf-8"):
        sentences = []
        with open(i_file_name, "r", encoding=i_enc) as f:
            for line in f:
                sentences.append(line.replace("\n", ""))
        print("Read:", i_file_name)
        return sentences



    ## ファイルに書き込む.
    #INPUT# o_file_name(String): 書き込むファイルのパス
    #INPUT# data(List_String): 書き込みたい内容
    #INPUT# o_enc(String): ファイル書き込み時の文字コード
    #RETURN# なし
    def write_file(self, o_file_name, data, o_enc="utf-8"):
        with open(o_file_name, "w", encoding=o_enc) as output:
            for line in data:
                output.write(str(line)+"\n")
        print("Write:", o_file_name)



    ## ディレクトリ内のファイルを検索する.
    #INPUT# dir_name(String): 検索したいディレクトリのパス
    #INPUT# condition(String): 正規表現で書かれた検索条件
    #INPUT# recursive(Boolean): 再帰的に検索するかどうか
    #INPUT# ignore(List_String): ファイル名に含まれていた場合に無視する文字列リスト
    #RETURN# files(List_String): 見つかったファイルのパス集合
    def search_files(self, dir_name, condition="/*", recursive=False, ignore=[]):
        files     = []
        name_list = glob.glob(dir_name+condition)
        for name in name_list:
            if name.split("/")[-1] in ignore:
                continue
            if os.path.isfile(name):
                files.append(name)
            if recursive and os.path.isdir(name):
                files.extend(self.search_files(name, condition, recursive=True))
        print("Find", len(files), "files in", dir_name)
        return files



    ## ディレクトリ内のディレクトリを検索する.
    #INPUT# dir_name(String): 検索したいディレクトリのパス
    #INPUT# condition(String): 正規表現で書かれた検索条件
    #INPUT# ignore(List_String): ディレクトリ名に含まれていた場合に無視する文字列リスト
    #RETURN# dirs(List_String): 見つかったディレクトリのパス集合
    def search_directories(self, dir_name, condition="*", ignore=[]):
        dirs = []
        name_list = glob.glob(dir_name+condition)
        for name in name_list:
            if name.split("/")[-1] in ignore:
                continue
            if os.path.isdir(name):
                    dirs.append(name)

        print("Find", len(dirs), "directories in", dir_name)
        return dirs



    ## ディレクトリを作成する.
    #INPUT# dir_name(String): 作成するディレクトリ名
    #RETURN# なし
    def make_directory(self, dir_name):
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
            print("create directory:", dir_name)



    ## python が実行されているディレクトリの絶対パスを取得する.
    #RETURN# *(String): python が実行されているディレクトリの絶対パス
    def get_current_path(self):
        return os.getcwd()



    ## ファイルが存在するかを調べる.
    #INPUT# file_name(String): 調べたいファイルのパス
    #RETURN# *(Boolean): ファイルが存在するかどうか
    def check_file_existence(self, file_name):
        return os.path.isfile(file_name)



## テスト用のメイン関数
if __name__ == "__main__":
    arranger = FileArranger()

    input_file  = "test_downloader"
    output_file = "test_downloader_s"
    arranger.segment_file(input_file, output_file, is_lemma=False)
