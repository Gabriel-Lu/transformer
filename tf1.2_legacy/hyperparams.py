# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
class Hyperparams:
    '''Hyperparameters'''
    # data
    # Tabriel~改这里
    source_train = 'raw_data/train_query.json'
    target_train = 'raw_data/train_response.json'
    source_test = 'raw_data/test_query.json'
    target_test = 'raw_data/test_response.json'
    
    # training
    batch_size = 8 # alias = N. ORIGIN :32
    lr = 0.0001 # 初始学习速率learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir' # log directory
    
    # model
    maxlen = 300 #一句话里最多可以有100个词语
                # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 60 #出现次数少于20次的就会被当作UNK来处理
                # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512 #隐藏节点的个数为512
                        # alias = C
    num_blocks = 6 # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False # If True, use sinusoid. If false, positional embedding.
    
    
    
    
