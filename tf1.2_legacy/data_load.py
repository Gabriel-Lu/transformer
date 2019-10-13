# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
包含了所有关于加载数据和批量化数据的函数
'''
from __future__ import print_function
from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import regex

'''
给de的每个词分配一个id并返回两个词典
一个词典是根据词找id,另一个是根据id找词
    # 直接利用codexs的open来读取之前预处理的时候生成的词汇文件。注意，这里去掉了那些次数少于min_cnt的词汇
    # 读完之后生成一个词汇列标，根据列标的枚举，生成词和id的两个字典
'''
def load_de_vocab():

    vocab = [line.split()[0] for line in codecs.open('preprocessed/de.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word
'''
给en的每个词分配一个id并返回两个词典
具体操作同上
'''
def load_en_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/en.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

'''
参数: source_sents可以理解为源语言的句子列表，列表中的每个元素就是一个句子
      target_sents同理

'''
def create_data(source_sents, target_sents): 
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    
    # Index
    x_list, y_list, Sources, Targets = [], [], [], []
    # 遍历句子列表。一次遍历一个句子对
    # 在本次遍历中，给每个句子的末尾都添加一个文本结束符/s用以表示句子结尾
    for source_sent, target_sent in zip(source_sents, target_sents):
        x = [de2idx.get(word, 1) for word in (source_sent + u" </S>").split()] # 1: OOV, </S>: End of Text
        y = [en2idx.get(word, 1) for word in (target_sent + u" </S>").split()] 
        if max(len(x), len(y)) <=hp.maxlen:
            # 利用双语的word/id字典读取word对应的id加入到新列表中。若word不在字典中，则id用1代替。
            # 将生成两个用一串id表示的双语句子的列标。
            # 若这两个句子的长度都没有超过限定hp.maxlen,则把他们的id列标加入到模型要用的id列表x_list和y_list中
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            # 把满足max句子长度的原始句子（用word表示的）也加入到句子列表Sources和Target中
            Sources.append(source_sent)
            Targets.append(target_sent)
    
    # Pad操作 在原来的数组上对维度进行扩展      
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, hp.maxlen-len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen-len(y)], 'constant', constant_values=(0, 0))
    
    return X, Y, Sources, Targets
'''
加载训练数据
加载方式：加载create_data返回的等长句子id数组
    # de_sents和en_sents这两个句子列表是通过读取训练数据生成的。读取过后按照换行符分隔开每一句
    # 在这些句子中选择那些开头符号不是'<'的句子
    # 在这些分离好的句子中同样使用正则表达式处理
'''
def load_train_data():
    de_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in codecs.open(hp.source_train, 'r', encoding='utf-8',errors='ignore').read().split("\n") if line and line[0] != "<"]
    en_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in codecs.open(hp.target_train, 'r', encoding='utf-8',errors='ignore').read().split("\n") if line and line[0] != "<"]
    '''
    print("/n/n/n/n/n/n/n/n/n/nPRINTING DE_SENTS NOW")
    for i in de_sents:
        print(str(i))
    
    print("/n/n/n/n/n/n/n/n/n/nPRINTING EN_SENTS NOW")
    for i in en_sents:
        print(str(i))
    '''
    X, Y, Sources, Targets = create_data(de_sents, en_sents)
    X, Y, Sources, Targets = create_data(de_sents, en_sents)
    
    return X, Y
'''
加载测试数据
跟load_train_data区别不大，目的是生成测试数据源语言的id表示的长句子列表。（而目标语言由模型来预测，所以不用生成）

'''
def load_test_data():
    def _refine(line):
        line = regex.sub("<[^>]+>", "", line)
        line = regex.sub("[^\s\p{Latin}']", "", line) 
        return line.strip()     # strip函数默认是去掉字符串首尾的空白符
    # 不同于Load_train_data,这里我们认为文件中每行以< seg 开头的才是真正训练数据的句子
    de_sents = [_refine(line) for line in codecs.open(hp.source_test, 'r', 'utf-8').read().split("\n") if line and line[:4] == "<seg"]
    en_sents = [_refine(line) for line in codecs.open(hp.target_test, 'r', 'utf-8').read().split("\n") if line and line[:4] == "<seg"]
        
    X, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X, Sources, Targets # (1064, 150)
'''
用于生成一个batch数据
'''
def get_batch_data():
    # 加载训练数据：源语言和目标语言id表示的句子数组
    X, Y = load_train_data()
    print("\n\n\n\n\n\n\n\n\n\nFIND ERROR")
    print("length of X is: "+str(len(X))+"\n")
    print("length of batch_size is: "+str(hp.batch_size)+"\n")

    # calc total batch count
    num_batch = len(X) // hp.batch_size
    print("num_batch ="+str(len(X))+" // "+str(hp.batch_size)+" is:"+str(num_batch))
    
    # 将两个数组转化为tensorflow支持的tensor这种数据类型
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)
    
    # Create Queues
    # 用函数对训练数据进行处理。从输入中每次去一个切片返回到输入队列，该队列作为之后tf.train.shuffle_batch的一个参数，以生成一个batch数据
    input_queues = tf.train.slice_input_producer([X, Y])
            
    # create batch queues
    x, y = tf.train.shuffle_batch(input_queues,
                                num_threads=8,
                                batch_size=hp.batch_size, 
                                capacity=hp.batch_size*64,   
                                min_after_dequeue=hp.batch_size*32, 
                                allow_smaller_final_batch=False)
    
    return x, y, num_batch # (N, T), (N, T), ()

