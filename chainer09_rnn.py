
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, \
                    optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

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

train_data = load_data('ptb.train.txt')
eos_id = vocab['<eos>']

# define model

class MyRNN(chainer.Chain):
    def __init__(self, v, k):
        super(MyRNN, self).__init__(
            embed = L.EmbedID(v, k),    #単語数v, 分散表現k次元
            H = L.Linear(k, k),         #k*kのCNN
            W = L.Linear(k, v),         #k*vのCNN
        )
        print ("Init MyRNN. v={0}, k={1}".format(v, k))
    def __call__(self, s): #文sを入力して誤差計算
        accum_loss = None
        v, k = self.embed.W.data.shape  #単語数v * 分散表現k
        h = Variable(np.zeros((1,k), dtype=np.float32)) #CNNの入力
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

# initialize model

demb = 100
model = MyRNN(len(vocab), demb)
optimizer = optimizers.Adam()
optimizer.setup(model)

# learn and save

for epoch in range(3):
    s = []
    for pos in range(len(train_data)): # コーパス
        id = train_data[pos]
        s.append(id)
        if (id == eos_id): # 文の最後になったら重みを更新
            loss = model(s)
            model.zerograds()
            loss.backward()
            optimizer.update()
            s = []
        if (pos % 1000 ==0):
            print (pos, "/", len(train_data), " finished")
    outfile = "myrnn-" + str(epoch) + ".model"
    serializers.save_npz(outfile, model)
            

