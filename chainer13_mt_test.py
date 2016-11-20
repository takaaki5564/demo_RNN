## Encoder-Decode transcription model

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, \
    optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import MeCab

# Generate Japanese words dictionary & word_id
jvocab = {}
jlines = open('jp2.txt').read().split('\n')
for i in range(len(jlines)):
    lt = jlines[i].split()
    for w in lt:
        if w not in jvocab:
            jvocab[w] = len(jvocab)
jvocab['<eos>'] = len(jvocab)
jv = len(jvocab)

# Generate English words dictionary & word_id
evocab = {}
id2wd = {}
elines = open('eng2.txt').read().split('\n')
for i in range(len(elines)):
    lt = elines[i].split()
    for w in lt:
        if w not in evocab:
            val = len(evocab)
            id2wd[val] = w
            evocab[w] = val
            
val = len(evocab)
id2wd[val] = '<eos>'
evocab['<eos>'] = val
ev = len(evocab)

class MyMT(chainer.Chain):
    def __init__(self, jv, ev, k):
        super(MyMT, self).__init__(
            embedx = L.EmbedID(jv, k),
            embedy = L.EmbedID(ev, k),
            H = L.LSTM(k, k),   # LSTM output -> history
            W = L.Linear(k, ev) # total output
        )
    def __call__(self, jline, eline): # not called in this code
        for i in range(len(jline)):
            wid = jvocab[jline[i]]
            x_k = self.embedx(Variable(np.array([wid], dtype=np.int32)))
            h = self.H(x_k)     # LSTM output -> no use
        # end of jp sentence = <eos>
        x_k = self.embedx(Variable(np.array([jvocab['<eos>']], dtype=np.int32)))  
        # LSTM output of jp <eos> = 1st word of eng sentence      
        tx = Variable(np.array([evocab[eline[0]]], dtype=np.int32))
        h = self.H(x_k)
        accum_loss = F.softmax_cross_entropy(self.W(h), tx)
        for i in range(1,len(eline)):
            wid = evocab[elin[i]]
            x_k = self.embedy(Variable(np.array([wid], dtype=np.int32))) 
            # training data of output of eng word = next word
            next_wid = evocab['<eos>'] if (i==len(eline)-1) else evocab[eline[i+1]]
            tx = Variable(np.array([next_wid], dtype=np.int32))
            h = self.H(x_k)
            loss = F.softmax_cross_entropy(self,W(h), tx)
            accum_loss = loss if accum_loss is None else accum_loss+loss
        print ("loss=", accum_loss)
        return accum_loss   # return loss

def to_words(sentence):
    tagger = MeCab.Tagger('mecabrc')
    mecab_result = tagger.parse(sentence)
    info_of_words = mecab_result.split('\n')
    
    words = []
    for info in info_of_words:
        if info == 'EOS' or info == '':
            break
        info_elems = info.split(',')
        print(info_elems)
        #print(info_elems[6])
        if info_elems[6] == '*':
            words.append(info_elems[0][:-3])
            continue
        words.append(info_elems[6])
    return tuple(words)

# evaluate model
def mt(model ,jline):       # jp sentence (test data)
    for i in range(len(jline)):
        if jline[i] in jvocab:
            wid = jvocab[jline[i]]
            x_k = model.embedx(Variable(np.array([wid],dtype=np.int32),volatile='on'))
            h = model.H(x_k)    # LSTM output of jp word -> no use
    # input jp <eos>
    x_k = model.embedx(\
        Variable(np.array([jvocab['<eos>']],dtype=np.int32),volatile='on'))
    # output of jp <eos> is start of eng sentence
    h = model.H(x_k)
    wid = np.argmax(F.softmax(model.W(h)).data[0]) # most probable result
    if wid in id2wd:
        print (id2wd[wid],end=" ") # 
    else:
        print (wid,end=" ")
    loop  = 0
    while (wid != evocab['<eos>']) and (loop <= 30):
        # LSTM output of previous eng words become translated sentence
        x_k = model.embedy(Variable(np.array([wid],dtype=np.int32), volatile='on'))
        h = model.H(x_k)
        wid = np.argmax(F.softmax(model.W(h)).data[0]) # most probable result
        if wid in id2wd:
            print (id2wd[wid],end=" ") # transport wid->word
        else:
            print (wid,end=" ")
        loop += 1
    print()

jlines = open('jp_test.txt').read().split('\n')

word = []
demb = 100
for epoch in range(100):
    model = MyMT(jv, ev, demb)
    filename = "mt-" + str(epoch) + ".model"
    serializers.load_npz(filename, model)
    #print("len(jlines)=", len(jlines))
    for i in range(len(jlines)-1):
        print (jlines[i], "->")
        jln = jlines[i].split()
        word.append(to_words(jlines[i]))
        #print("input words=", word)
        #jlnr = jln[::-1]           # read from reverse (effective)
        word = word[::-1]
        print (epoch, ":",end="")
        #mt(model, jlnr)
        mt(model, word)


