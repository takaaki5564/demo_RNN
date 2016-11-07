
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, \
                    optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

import math
import sys
argvs = sys.argv

# set data

vocab = {}

def load_data(filename):
    global vocab
    # 改行を<eos>にする
    words = open(filename).read().replace('\n','<eos>').strip().split()
    dataset = np.ndarray((len(words),), dtype = np.int32) # 単語idの配列
    for i, word in enumerate(words): #indexと値を同時に返す
        if word not in vocab:
            vocab[word] = len(vocab) #コーパスの単語を順にvocabに登録
        dataset[i] = vocab[word]
    return dataset

# define model

class MyRNN(chainer.Chain):
    def __init__(self, v, k):
        super(MyRNN, self).__init__(
            embed = L.EmbedID(v, k),    # 単語数v, 分散表現k次元
            H = L.Linear(k, k),         # k*kのCNN
            W = L.Linear(k, v),         # k*vのCNN
        )
    def __call__(self, s): # 文sを入力して誤差計算
        accum_loss = None
        v, k = self.embed.W.data.shape  # 単語数v * 分散表現k
        h = Variable(np.zeros((1,k), dtype=np.float32)) # CNNの入力
        for i in range(len(s)):
            # i+1番目の単語s[i+1], 文の最後がeos
            next_w_id = eos_id if (i==len(s)-1) else s[i+1] 
            tx = Variable(np.array([next_w_id], dtype=np.int32))
            # i番目の単語s[i]が入力
            x_k = self.embed(Variable(np.array([s[i]], dtype=np.int32)))
            # i番目の単語s[i]と過去の記憶H, 中間層の入力h
            h = F.tanh(x_k + self.H(h))
            # 出力W(h)と次の単語txの誤差を計算
            loss = F.softmax_cross_entropy(self.W(h), tx)
            accum_loss = loss if accum_loss is None else accum_loss+loss
        return accum_loss

train_data = load_data('ptb.train.txt')

# initialize model

demb = 100 # 分散表現の次元数

def cal_ps(model, s):
    h = Variable(np.zeros((1,demb), dtype=np.float32)) # 中間値入力
    sum = 0.0
    for i in range(1,len(s)):
        w1, w2 = s[i-1], s[i] # ひと続きの単語
        x_k = model.embed(Variable(np.array([w1], dtype=np.int32))) # 入力信号
        h = F.tanh(x_k + model.H(h)) # RNN中間層の出力h
        yv = F.softmax(model.W(h))   # hを線形作用素Wをかける
        pi = yv.data[0][w2]
        sum -= math.log(pi, 2)
    return sum

eos_id = vocab['<eos>']
max_id = len(vocab)
print ("eos_id={0}, max_id={1}".format(eos_id, max_id))

model = MyRNN(len(vocab), demb)
#model = MyRNN(max_id, demb) # train時の単語数でモデルを初期化する
serializers.load_npz(argvs[1], model)
sum = 0.0
wnum = 0
s = []
unk_word = 0

test_data = load_data('ptb.test.txt') #loadする順番を変更
test_data = test_data[0:1000]

print ("len(vocab)={0}".format(len(vocab)))


for pos in range(len(test_data)):
    id = test_data[pos]
    s.append(id)
    if (id > max_id):
        unk_word = 1
    if (id == eos_id):
        if (unk_word != 1):
            ps = cal_ps(model, s)
            sum += ps
            wnum += len(s) - 1
        else:
            unk_word = 0
        s = []
print (math.pow(2, sum / wnum))

