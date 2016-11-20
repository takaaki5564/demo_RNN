## Encoder-Decoder Translator model
# -*- coding: utf-8 -*-

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, \
                    optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

# Generate Japanese Dictionary & Word ID
jvocab = {}
jlines = open('jp2.txt').read().split('\n')
for i in range(len(jlines)):
    lt = jlines[i].split()
    for w in lt:
        if w not in jvocab:
            jvocab[w] = len(jvocab)
            #print(len(jvocab),":", w, ", ", end="")

jvocab['<eos>'] = len(jvocab)   # end of all sentence
jv = len(jvocab)
print("num of vocab_jp: {0}".format(jv))

# Generate English Dictionary & Word ID
evocab = {}
elines = open('eng2.txt').read().split('\n')
for i in range(len(elines)):
    lt = elines[i].split()
    for w in lt:
        if w not in evocab:
            evocab[w] = len(evocab)
            #print(len(jvocab),":", w, ", ", end="")

evocab['<eos>'] = len(evocab)   # end of all sentence
ev = len(evocab)
print("num of vocab_eng: {0}".format(ev))

class MyMT(chainer.Chain):
    def __init__(self, jv, ev, k):      # vocab num Japanese(jv), English(ev)
        super(MyMT, self).__init__(
            embedx = L.EmbedID(jv, k),  # k次元分散表現
            embedy = L.EmbedID(ev, k),  # k次元分散表現
            H = L.LSTM(k, k),
            W = L.Linear(k, ev),
        )
    def __call__(self, jline, eline):
        #self.H.reset_state()
        for i in range(len(jline)):
            wid = jvocab[jline[i]]      # jword to id
            x_k = self.embedx(Variable(np.array([wid], dtype=np.int32)))
            h = self.H(x_k)             # LSTM output -> not use in jp
        # set <eos> at end of jp sentence
        x_k = self.embedx(Variable(np.array([jvocab['<eos>']],dtype=np.int32)))
        # 日本語の<eos>が英語の文頭になる
        tx =  Variable(np.array([evocab[eline[0]]], dtype=np.int32))
        h = self.H(x_k)                 # LSTM output of <eos> = first eng word
        #print("x_k, tx, h =", x_k.data, tx.data, h.data)
        accum_loss = F.softmax_cross_entropy(self.W(h), tx)
        # start calc english translation after japanese <eos>+1
        for i in range(len(eline)):
            wid = evocab[eline[i]]
            x_k = self.embedy(Variable(np.array([wid], dtype=np.int32)))
            next_wid = evocab['<eos>'] if (i==len(eline)-1) else evocab[eline[i+1]]
            tx = Variable(np.array([next_wid], dtype=np.int32)) # next word = train data
            h = self.H(x_k)             # LSTM output
            loss = F.softmax_cross_entropy(self.W(h), tx)   # calc error
            accum_loss += loss
        return accum_loss

demb = 100
model = MyMT(jv, ev, demb)
optimizer = optimizers.Adam()
optimizer.setup(model)

for epoch in range(100):
    sum_loss = 0.0
    for i in range(len(jlines)-1):
        jln = jlines[i].split() # 日本語文のi番目
        jlnr = jln[::-1]        # 逆方向に読み込む
        eln = elines[i].split() # 英語文のi番目
        model.H.reset_state()   # LSTMの状態リセット
        model.zerograds()       # 勾配リセット
        loss = model(jlnr, eln) # 日本語文、英語文を入力して誤差計算
        loss.backward()         # 逆伝搬
        loss.unchain_backward() # 計算履歴を後ろから切る
        optimizer.update()
        sum_loss += loss
        #print ("loss: ", loss.data)
    outfile = "mt-" + str(epoch) + ".model"
    serializers.save_npz(outfile, model)
    print("epoch[", epoch, "] model saved.(sum loss=", sum_loss.data, ")")





