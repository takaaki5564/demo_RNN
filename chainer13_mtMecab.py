#!/usr/bin/python
# -*- coding: utf-8 -*-
# 日本語を使う場合はutf-8の文字コードを記載

## Learning Encode-Decode Translation Model
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
        x_train = Variable(np.array([e_word2id[e_sentence[0]]],dtype=np.int32))
        h = self.H(x_dim)
        # loss of 1st word
        accum_loss = F.softmax_cross_entropy(self.W(h),x_train)
        # loss of 2nd or later words
        for i in range(len(e_sentence)):
            wid = e_word2id[e_sentence[i]]
            x_dim = self.embedy(Variable(np.array([wid],dtype=np.int32)))
            wid_next = e_word2id['<eos>']\
                if (i==len(e_sentence)-1) else e_word2id[e_sentence[i+1]]
            x_train = Variable(np.array([wid_next],dtype=np.int32))
            h = self.H(x_dim)
            # calc loss
            loss = F.softmax_cross_entropy(self.W(h),x_train)
            accum_loss += loss
        return accum_loss

j_numlines, j_numvocab, j_word2id, j_id2word, j_sentence = load_data('jp2.txt')
e_numlines, e_numvocab, e_word2id, e_id2word, e_sentence = load_data('eng2.txt')

print("j_numlines=",j_numlines)
print("e_numlines=",e_numlines)
print("j_numvocab=",j_numvocab)
print("e_numvocab=",e_numvocab)
#print("j_word2id=",j_word2id)
#print("e_word2id=",e_word2id)

if (j_numlines != e_numlines):
    print ("[ERR] num lines should be same.","j_numlines","!=","e_numlines")
    
demb = 100
model = MyMT(j_numvocab, e_numvocab, demb)
optimizer = optimizers.Adam()
optimizer.setup(model)

jlines = {}
elines = {}
for i in range(len(j_sentence)-1):
    jlines[i] = to_words(j_sentence[i])
    elines[i] = to_words(e_sentence[i])
    
jline = []
eline = []
numepoch = 100

# learning
for epoch in range(numepoch):
    sum_loss = 0.0
    for i in range(len(j_sentence)-1):
        #jline = j_sentence[i].split()
        #jline = to_words(j_sentence[i])
        jline = jlines[i]
        #print("divided line=", jline)
        jliner = jline[::-1]
        #print("reversed line=", jliner)
        #eline = e_sentence[i].split()
        #eline = to_words(e_sentence[i])
        eline = elines[i]
        #print("divided line=", eline)
        
        model.H.reset_state()
        model.zerograds()
        loss = model(jliner, eline)
        loss.backward()
        loss.unchain_backward()
        optimizer.update()
        sum_loss += loss.data
        #print (i, " finished")
    
    if (epoch == numepoch-1)
        outfile = "mt-" + str(epoch) + ".model"
        serializers.save_npz(outfile, model)
    print ("epoch=",epoch,", sum_loss=",sum_loss) 

