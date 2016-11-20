## LSTMで文章を学習する(10と同じ)
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable,\
    optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

# グローバル変数:単語idの登録
vocab = {}

def load_data(filename):
    global vocab
    word = open(filename).read().replace('\n','<eos>').strip().split()
    # dataset:単語辞書配列の初期化
    dataset = np.ndarray((len(words),), dtype = np.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)
        dataset[i] = vocab[word]
    return dataset

train_data = load_data('ptb.train.txt')
eos_id = vocab['<eos>']

class MyLSTM(chainer.Chain):
    def __init__(self, v, k):
        super(MyLSTM, self).__init__(
            embed = L.EmbedID(v,k), # word2vecでv個の単語をk次元分散表現に変換
            Wz = L.Linear(k,k),     # RNN層
            Wi = L.Linear(k,k),     # Input層
            Wf = L.Linear(k,k),     # Forget層
            Wo = L.Linear(k,k),     # Output層
            Rz = L.Linear(k,k),     # 履歴入力への重みR
            Ri = L.Linear(k,k),
            Rf = L.Linear(k,k),
            Ro = L.Linear(k,k),
            W = L.Linear(k,v),
        )
    def __call__(self, s): # sは入力文章の単語id配列
        accum_loss = None
        # 入力単語をword2vecで縮約
        v, k = self.embed.W.data.shape
        h = Variable(np.zeros((1,k), dtype = np.float32)) # 履歴入力
        c = Variable(np.zeros((1,k), dtype = np.float32)) # 入力
        
        # 注目単語の次の単語を訓練データとして学習
        for i in range(len(s)):
            next_w_id = eos_is if (i==len(s)-1) else s[i+1]
            # 訓練データ
            tx = Variable(np.array([next_w_id], dtype = np.int32))
            # 入力文のうちの単語[i]
            x_k = self.embed(Variable(np.array([s[i]], dtype = np.int32)))
            
            # RNN層の計算
            z0 = self.Wz(x_k) + self.Rz(h)
            z1 = F.tan(z0)
            # Input層の計算
            i0 = self.Wi(x_k) + self.Ri(F.dropout(h))
            i1 = F.sigmoid(i0)
            # Forget層の計算
            f0 = self.Wf(x_k) + slef.Rf(F.dropout(h))
            f1 = F.sigmoid(f0)
            # MemoryCellに渡す値の計算
            c = i1 * z1 + f1 * c
            # LSTM出力層の計算
            o0 = slef.Wo(x,k) + self.Ro(h)
            o1 = F.sigmoid(o0)
            # 次の時刻に渡す履歴計算
            h = o1 * F.tanh(c)
            loss = F.softmax_cross_entropy(self.W(h), tx)
            accum_loss = loss if accum_loss is None else accum_loss + loss
        return accum_loss
            
demb = 100 # 分散表現の次元数
model = MyLSTM(len(vocab), demb)

optimizer = optimizers.Adam()
optimizer.setup(model)

for epoch in range(5):
    s = []
    for pos in range(len(train_data)):
        id = train_data[pos]
        s.append(id)
        # 文の最後になるまで単語をappend
        if (id == eos_id):
            # 勾配の初期化
            model.zerograds()
            # 順伝搬誤差の計算
            loss = model(s)
            # 誤差逆伝搬の計算
            loss.backward()
            if (len(s) > 29):
                # 文字数が30以上離れている場合は無視
                loss.unchain_backward()
            # 重みの更新
            optimizer.update()
            s = []
        if (pos % 100 == 0):
            print (pos, "/", len(train_data), " finished")
    outfile = "lstm1-" + str(epoch) + ".model"
    serialziers.save_npz(outfile, model)
    
            















