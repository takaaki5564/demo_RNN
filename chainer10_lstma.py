#LSTM改良版

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable,\
                    optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

vocab = {}

def load_data(filename):
    global covab
    words = open(filename).read().replace('\n','<eos>').strip().split()
    dataset = np.ndarray((len(words),), dtype=np.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = len (vocab)
        dataset[i] = vocab[word]
    return dataset

train_data = load_data('ptb.train.txt')
eos_id = vocab['<eos>']

class MyLSTM(chainer.Chain):
    def __init__(self, v, k):
        super(MyLSTM, self).__init__(
            embed = L.EmbedID(v, k),
            Wz = L.Linear(k, k),
            Wi = L.Linear(k, k),
            Wf = L.Linear(k, k),
            Wo = L.Linear(k, k),
            W = L.Linear(k, v),  #Historyの重みRをWと共通化
        )
    def __call__(self, s): #s:単語id配列
        accum_loss = None
        v, k = self.embed.W.data.shape
        h = Variable(np.zeros((1,k), dtype=np.float32)) #History
        c = Variable(np.zeros((1,k), dtype=np.float32)) #Input
        
        for i in range(len(s)):
            next_w_id = eos_id if (i==len(s)-1) else s[i+1]
            tx = Variable(np.array([next_w_id], dtype=np.int32)) #訓練データは次の単語id
            x_k = self.embed(Variable(np.array([s[i]], dtype=np.int32))) #k次分散表現
            
            z0 = self.Wz(x_k) + self.Wz(h) #RNN
            z1 = F.tanh(z0)
            i0 = self.Wi(x_k) + self.Wi(h) #Input Layer
            i1 = F.sigmoid(i0)
            f0 = self.Wf(x_k) + self.Wf(h) #Forget Layer
            f1 = F.sigmoid(f0)
            c = i1 * z1 + f1 * c
            o0 = self.Wo(x_k) + self.Wo(h)
            o1 = F.sigmoid(o0)
            h = o1 * F.tanh(c)
            loss = F.softmax_cross_entropy(self.W(h), tx)
            accum_loss = loss if accum_loss is None else accum_loss + loss
        return accum_loss

demb = 100
model = MyLSTM(len(vocab), demb)
optimizer = optimizers.Adam()
optimizer.setup(model)

for epoch in range(5):
    s = []
    for pos in range(len(train_data)):
        id = train_data[pos]
        s.append(id)
        if (id == eos_id):  #一分ずつ入れて重みを更新する
            model.zerograds()
            loss = model(s)
            loss.backward()
            if (len(s) > 29): #一文の中の30語以上は忘れる
                loss.unchain_backward()
            optimizer.update()
            s = []
        if (pos % 100 == 0):
            print ("{0}/{1} finished".format(pos, len(train_data)))
    outfile = "lstm0a-" + str(epoch) + ".model"
    serializers.save_npz(outfile, model)
            
