# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
生成源语言和目标语言的词汇文件
* 把训练数据中的单词和出现次数进行了统计，记录在生成的词汇文件中。文件中第一列为单词，第二列为出现的次数
* 同时设置了4个标记符号，PAD UNK S /S
'''
from __future__ import print_function
from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import os
import regex
from collections import Counter

def make_vocab(fpath, fname):
    '''生成词汇文件的函数
    
    Args:
      fpath: A string. 输入文件的路径，即训练数据
      fname: A string. 要输出的词汇文件的名字
    
    Writes vocabulary line by line to `preprocessed/fname`
    '''  
    text = codecs.open(fpath, 'r', encoding='utf-8',errors='ignore').read()
    text = regex.sub("[^\s\p{Latin}']", "", text)
    words = text.split()
    word2cnt = Counter(words)
    if not os.path.exists('preprocessed'): os.mkdir('preprocessed')
    # 一行一行的把词汇写入到preprocessed/fname中
    with codecs.open('preprocessed/{}'.format(fname), 'w', 'utf-8') as fout:
        fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write(u"{}\t{}\n".format(word, cnt))

if __name__ == '__main__':
    make_vocab(hp.source_train, "de.vocab.tsv")
    make_vocab(hp.target_train, "en.vocab.tsv")
    print("Done")