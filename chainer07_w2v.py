
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable,\
                    optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

from chainer.utils import walker_alias  # 離散型確率分布からの乱数生成アルゴリズム
import collections  # need for word2vec

# set data

index2word = {}
word2index = {}
counts = collections.Counter()
dataset = []
with open('ptb.train.txt') as f:
    for line in f:
        for word in line.split():
            if word not in word2index:      # add words to w2v dictionary
                ind = len(word2index)       # get last index
                word2index[word] = ind      # add word
                index2word[ind] = word      #
            counts[word2index[word]] += 1   # appearance counts of word
            dataset.append(word2index[word]) # add id to dataset

# datasetはコーパス内の全単語が登録されている
# コーパス内のn番目の単語のidがdataset[n]に登録されている
# id=kの単語はindex2word[k]で参照できる

n_vocab = len(word2index)
datasize = len(dataset)

print ("num of vocabrary : ", n_vocab)
print ("size of dataset  : ", datasize)

cs = [counts[w] for w in range(len(counts))]
power = np.float32(0.75)
p = np.array(cs, power.dtype)
sampler = walker_alias.WalkerAlias(p)       # サンプル生成器
# sampler.sample(5)とすれば、5つサンプリングされた単語id配列が返る

# define model

class MyW2V(chainer.Chain):                 # Chainクラスを継承
    def __init__(self, n_vocab, n_units):   # 単語数と
        super(MyW2V, self).__init__(
            # EmbedID: 引数(1)=単語の種類数, (2)=分散表現の次元数
            embed = L.EmbedID(n_vocab, n_units), #w2vでの次元縮約？
        )
    def __call__(self, xb, yb, tb): #(xb,yb):単語のIDのペア, tb:教師信号
        xc = Variable(np.array(xb, dtype=np.int32))
        yc = Variable(np.array(yb, dtype=np.int32))
        tc = Variable(np.array(tb, dtype=np.int32))
        return F.sigmoid_cross_entropy(self.fwd(xc,yc), tc) #順伝搬
    def fwd(self, x, y):
        x1 = self.embed(x)  # 単語id=xに対する分散表現x1
        x2 = self.embed(y)  # 単語id=yに対する分散表現x2
        return F.sum(x1 * x2, axis=1)  #ベクトルの内積
        
# my functions

ws = 3  # window size : 前後に注目する単語数
ngs = 5 # negative sample size : 負例の単語数

#単語idのペアのバッチを作る
def mkbatset(dataset, ids):
    xb, yb, tb = [], [], []
    for pos in ids:
        xid = dataset[pos]      #dataset=コーパスの単語を順に注目
        for i in range(1, ws):
            p = pos - i
            if p >= 0:
                xb.append(xid)      #着目する単語idをxbに追加
                yid = dataset[p]    #周辺の単語idをyidに単語
                yb.append(yid)      #周辺の単語idでバッチybを作成
                tb.append(1)        #教師信号バッチに1を追加
                for nid in sampler.sample(ngs): #負例からサンプリング
                    #負例は周辺単語と組み合わせて教師信号を0とする
                    xb.append(yid)  #周辺単語idをxbに追加
                    yb.append(nid)  #負例単語idをybに追加
                    tb.append(0)    #負例の教師信号は0
            p = pos + i
            if p < datasize:
                xb.append(xid)
                yid = dataset[p]
                yb.append(yid)
                tb.append(1)
                for nid in sampler.sample(ngs):
                    xb.append(yid)
                    yb.append(nid)
                    tb.append(0)
    return [xb, yb, tb]

# Initialize mode

demb = 100                      #分散表現ベクトル次元
model = MyW2V(n_vocab, demb)    #CNNモデル初期化
optimizer = optimizers.Adam()   #最適化はAdam
optimizer.setup(model)

# Learn

bs = 100

for epoch in range(5):
    print('epoch: {0}'.format(epoch))
    indexes = np.random.permutation(datasize) #datasize分の乱数index生成
    period = int(datasize/bs/10)
    for pos in range(0, datasize, bs):
        if (pos % period) == 0:
            print (epoch, pos)
        #バッチサイズ分の乱数idxを生成
        ids = indexes[pos:(pos+bs) if (pos+bs) < datasize else datasize]
        xb, yb, tb = mkbatset(dataset, ids) # バッチセットをランダムに作成
        model.zerograds()
        loss = model(xb, yb, tb)
        loss.backward()
        optimizer.update()

# save model

# pythonでファイル操作にwithを使うと便利
with open('w2v.model', 'w') as f:
    # 辞書登録数とバッチサイズ?を記録
    f.write('%d %d\n' % (len(index2word), 100))
    w = model.embed.W.data
    for i in range(w.shape[0]):
        # index=iの単語と
        v = ' '.join(['%f' % v for v in w[i]]) # join:文字列の連結
        f.write('%s %s\n' % (index2word[i], v)) # 単語とベクトルを書き込み
       
