
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable,\
                    optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

from chainer.utils import walker_alias
import collections

# set data

index2word = {}
word2index = {}
counts = collections.Counter()
dataset = []

with open('ptb.train.txt') as f:
    for line in f:
        for word in line.split():
            if word not in word2index:
                ind = len(word2index)
                word2index[word] = ind
                index2word[ind] = word
            counts[word2index[word]] = +1
            dataset.append(word2index[word])

n_vocab = len(word2index)
datasize = len(dataset)
print("num_of_vocab, datasize : ({0}, {1})".format(n_vocab, datasize))

cs = [counts[w] for w in range(len(counts))]
power = np.float32(0.75)
p = np.array(cs, power.dtype) # ネガティブサンプル生成器(確率分布)
sampler = walker_alias.WalkerAlias(p)

# define model

class MyW2V2(chainer.Chain): # Chainクラスを継承
    def __init__(self, v, m):
        super(MyW2V2, self).__init__( # superクラスの初期化を継承
            embed = L.EmbedID(v,m),   # 単語数vの分散表現次元m
        )
    def __call__(self, xb, eb, sampler, ngs):
    # xb:単語, eb:分散表現, sampler:sample生成器, ngs:負例数
        loss = None
        for i in range(len(xb)):
            x = Variable(np.array([xb[i]], dtype=np.int32))
            e = eb[i]
            ls = F.negative_sampling(e, x, self.embed.W, sampler, ngs)
            loss = ls if loss is None else loss + ls
        return loss

# my functions

ws = 3  # window size
def mkbatset(model, dataset, ids):
    xb, eb = [], []
    for pos in ids: # ids=バッチ処理するid:登録した単語を順に注目
        xid = dataset[pos]
        for i in range(1,ws):
            p = pos - i
            if p >= 0: # xidの前、3単語分のid=pに着目
                xb.append(xid)
                eid = dataset[p] # 正例のid登録
                eidv = Variable(np.array([eid], dtype=np.int32)) # Variableに登録
                ev = model.embed(eidv)  # 分散表現を取得
                eb.append(ev)
            p = pos + i
            if p < datasize: # xidの後、3単語分のid=pに着目
                xb.append(xid)
                eid = dataset[p]
                eidv = Variable(np.array([eid], dtype=np.int32))
                ev = model.embed(eidv)
                eb.append(ev)
    return [xb, eb] # 単語idと正例の分散表現ベクトルの組み合わせを返す

# intialize model
dim = 100
model = MyW2V2(n_vocab, dim) # 100:分散表現の次元数
optimizer = optimizers.Adam()
optimizer.setup(model)

# learn

bs = 50
ngs = 5

for epoch in range(10):
    print('epoch: {0}'.format(epoch))
    indexes = np.random.permutation(datasize)
    for pos in range(0, datasize, bs):
        print (epoch, pos)
        ids = indexes[pos:(pos+bs) if (pos+bs) < datasize else datasize]
        xb, eb = mkbatset(model, dataset, ids)
        model.zerograds()
        loss = model(xb, eb, sampler.sample, ngs)
        loss.backward()
        optimizer.update()

# save model

with open('w2v2.model', 'w') as f:
    f.write('%d %d\n' % (len(index2word), dim))
    w = model.lf.W.data
    for i in range(w.shape[0]):
        v = ' '.join(['%f' % v for v in w[i]])
        f.write('%s %s\n' % (index2word[i], v))


