# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
评估模型的效果
'''

from __future__ import print_function
import codecs
import os

import tensorflow as tf
import numpy as np

from hyperparams import Hyperparams as hp
from data_load import load_test_data, load_de_vocab, load_en_vocab
from train import Graph
from nltk.translate.bleu_score import corpus_bleu   #是NLTK里面方便计算翻译效果bleu score的模块

def eval(): 
    # Load graph
    g = Graph(is_training=False)
    print("Graph loaded")
    
    # Load data
    X, Sources, Targets = load_test_data()
    de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
     
#     X, Sources, Targets = X[:33], Sources[:33], Targets[:33]
     
    # Start session         
    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ## Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")
              
            ## Get model name
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1] # model name
             
            ## Inference
            if not os.path.exists('results'): os.mkdir('results')
            with codecs.open("results/" + mname, "w", "utf-8") as fout:
                print("open reults success\n")
                list_of_refs, hypotheses = [], []
                print("length of batch is"+str(len(X) // hp.batch_size))
                for i in range(len(X) // hp.batch_size):
                    print('translating')
                    ### Get mini-batches
                    x = X[i*hp.batch_size: (i+1)*hp.batch_size]
                    sources = Sources[i*hp.batch_size: (i+1)*hp.batch_size]
                    targets = Targets[i*hp.batch_size: (i+1)*hp.batch_size]
                     
                    ### Autoregressive inference
                    # 循环结束后，这个batch的句子的翻译保存在preds中
                    preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
                    for j in range(hp.maxlen):
                        _preds = sess.run(g.preds, {g.x: x, g.y: preds})
                        preds[:, j] = _preds[:, j]
                     
                    ### Write to file
                    # 翻译完成后把句子的翻译保存到preds中
                    for source, target, pred in zip(sources, targets, preds): # sentence-wise
                        #将pred(preds中的每个句子)的每个id转化为其对应的而英文单词，然后将这些单词字符串用一个空格字符链接起来，同时去掉句尾结束符。即得到了翻译的由词组成的句子。
                        got = " ".join(idx2en[idx] for idx in pred).split("</S>")[0].strip()    
                        # 分别将原句子、期望翻译的结果、实际翻译的结果写入文件
                        fout.write("- source: " + source +"\n")
                        print('\n'+'\n'+'\n'+source+'\n'+'\n'+'\n')
                        fout.write("- expected: " + target + "\n")
                        fout.write("- got: " + got + "\n\n")
                        fout.flush()
                          
                        # bleu score
                        
                        ref = target.split()
                        hypothesis = got.split()
            

                        if len(ref) > 3 and len(hypothesis) > 3:
                            list_of_refs.append([ref])
                            hypotheses.append(hypothesis)
              
                ## Calculate bleu score
                # 最后计算bleu score并写入文件
                # 将两者长度都大于3的句子加入到总的列表中，作为计算Bleu的参数，由此得到bleu socre.可以用以评估模型。
                str_hyp=",".join(hypotheses)
                print("len of hypothese is :"+len(hypotheses))
                score = corpus_bleu(list_of_refs, hypotheses)
                fout.write("Bleu Score = " + str(100*score))    #将bleu score写入文件末尾
                                          
if __name__ == '__main__':
    eval()
    print("Done")
    
    