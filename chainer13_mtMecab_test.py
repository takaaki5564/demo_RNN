#!/usr/bin/python
# -*- coding: utf-8 -*-
# 日本語を使う場合はutf-8の文字コードを記載

## Evaluate Encode-Decode Translation Model
## Japanese to English

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable,\
    optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import MeCab

def to_words(sentence):
    """
    入力: 'すべて自分のほうへ'
    出力: tuple(['すべて', '自分', 'の', 'ほう', 'へ'])
    """
    tagger = MeCab.Tagger('mecabrc')
    mecab_result = tagger.parse(sentence)
    info_of_words = mecab_result.split('\n')
    
    words = []
    for info in info_of_words:
        # macabで分けると、文の最後に’’が、その手前に'EOS'が来る
        if info == 'EOS' or info == '':
            break
            # info => 'な\t助詞,終助詞,*,*,*,*,な,ナ,ナ'
        info_elems = info.split(',')
        #print(info_elems)              # MeCab analysis result
        # 6番目に、無活用系の単語が入る。もし6番目が'*'だったら0番目を入れる
        if info_elems[6] == '*':
            # info_elems[0] => 'ヴァンロッサム\t名詞'
            words.append(info_elems[0][:-3])
            continue
        words.append(info_elems[6])
    #return tuple(words)
    return words
    
def load_data(fname):
    word2id = {}
    id2word = {}
    sentence = open(fname).read().split('\n')
    #print("open file=", fname)
    for i in range(len(sentence)):
        words = []
        #print("sentence[i]=", sentence[i])
        words = to_words(sentence[i])
        #print("words=", words)
        for word in words:
            if word not in word2id:
                #print("add word=", word)
                idx = len(word2id)
                id2word[idx] = word
                word2id[word] = idx
    idx = len(word2id)
    id2word[idx] = '<eos>'
    word2id['<eos>'] = idx
    numlines = len(sentence)
    numvocab = len(word2id)
    return numlines, numvocab, word2id, id2word, sentence

j_numlines, j_numvocab, j_word2id, j_id2word, j_sentence = load_data('jp2.txt')
e_numlines, e_numvocab, e_word2id, e_id2word, e_sentence = load_data('eng2.txt')

print("j_numlines=",j_numlines)
print("e_numlines=",e_numlines)
print("j_numvocab=",j_numvocab)
print("e_numvocab=",e_numvocab)
#print("j_word2id=",j_word2id)
#print("e_word2id=",e_word2id)

class MyMT(chainer.Chain):
    def __init__(self, j_numvocab, e_vocab, dim):
        super(MyMT, self).__init__(
            embedx = L.EmbedID(j_numvocab, dim),    # jp input-x
            embedy = L.EmbedID(e_numvocab, dim),    # eng input-y
            H = L.LSTM(dim, dim),           # LSTM output (recurrent part)
            W = L.Linear(dim, e_numvocab)   # Total output
        )
        
    def __call__(self, j_sentence, e_sentence):
        for i in range(len(j_sentence)):
            wid = j_word2id[j_sentence[i]]
            x_dim = self.embedx(Variable(np.array([wid],dtype=np.int32)))
            h = self.H(x_dim)
        x_dim = self.embedx(Variable(np.array([j_word2id['<eos>']],dtype=np.int32)))
        x_train = Variabl(np.array([e_word2id[e_sentence[0]]],dtype=np.int32))
        h = self.H(x_dim)
        # loss of 1st word
        accum_loss = F.softmax_cross_entropy(self.W(h),x_train)
        # loss of 2nd or later words
        for i in range(1,len(e_sentence)):
            wid = e_word2id[e_sentence[i]]
            x_dim = self.embedy(Variable(np.array([wid],dtype=np.int32)))
            wid_next = e_word2id['<eos>'] \
                if (i==len(e_sentence)-1) else e_word2id[e_sentence[i+1]]
            x_train = Variable(np.array([next_wid],dtype=np.int32))
            h = self.H(x_dim)
            # calc loss
            loss = F.softmax_cross_entropy(self.W(h),x_train)
            accum_loss = loss if accum_loss is None else accum_loss+loss
        
        return accum_loss

def evalmodel(model, j_sentence):
    for i in range(len(j_sentence)):
        if j_sentence[i] in j_word2id:
            wid = j_word2id[j_sentence[i]]
            x_dim = model.embedx(Variable(np.array([wid],dtype=np.int32),volatile='on'))
            h = model.H(x_dim)      # LSTM output -> no use
    x_dim = model.embedx(Variable(\
            np.array([j_word2id['<eos>']],dtype=np.int32),volatile='on'))
    h = model.H(x_dim)          # LSTM output -> start use
    
    # output argmax word
    wid = np.argmax(F.softmax(model.W(h)).data[0])
    if wid in e_id2word:
        print (e_id2word[wid],end=" ")
    else:
        print (wid, end=" ")
    
    loop = 0
    while (wid != e_word2id['<eos>']) and (loop<=30):   # sentence less than 30 words
        x_dim = model.embedy(Variable(\
                np.array([wid],dtype=np.int32), volatile='on'))
        h = model.H(x_dim)
        
        # output argmax word
        wid = np.argmax(F.softmax(model.W(h)).data[0])
        if wid in e_id2word:
            print (e_id2word[wid],end=" ")
        else:
            print (wid,end=" ")
        loop += 1
    print ()    # new line
    
jt_numlines, jt_numvocab, jt_word2id, jt_id2word, jt_sentence = load_data('jp_test.txt')

print("jt_numlines=",jt_numlines)
print("jt_numvocab=",jt_numvocab)
#print("jt_word2id=",jt_word2id)

jtlines = {}
for i in range(len(jt_sentence)-1):
    jtlines[i] = to_words(jt_sentence[i])
    
jtline = []

demb = 100
numepoch = 100
#for epoch in range(numepoch):
for epoch in [numepoch-1]:
    model = MyMT(j_numvocab, e_numvocab, demb)
    filename = "mt-" + str(epoch) + ".model"
    serializers.load_npz(filename, model)
    for i in range(len(jt_sentence)-1):
        jtline = jtlines[i]
        jtliner = jtline[::-1]
        print ("epoch=", epoch, end=", ")
        evalmodel(model, jtliner)











