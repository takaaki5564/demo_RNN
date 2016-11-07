import numpy
import six

# num of search result to show
n_result = 5

# .modelファイルには一行目に[単語数,次元数]
# 二行目以降に単語, ベクトル要素が記載されている

with open('w2v.model', 'r') as f:
    # 空白で単語を分割
    ss = f.readline().split()
    # 単語数, 分散表現次元数
    n_vocab, n_units = int(ss[0]), int(ss[1])
    word2index = {}
    index2word = {}
    w = numpy.empty((n_vocab, n_units), dtype=numpy.float32)
    
    for i, line in enumerate(f):
        ss = line.split()
        assert len(ss) == n_units + 1   # ssの長さは次元数+1(単語)
        word = ss[0]                    # ss[0]=単語
        word2index[word] = i            # i=行数=単語index
        index2word[i] = word            # 再度indexの単語を登録
        # 分散表現ベクトルをfloatでw[i]に格納
        w[i] = numpy.array([float(s) for s in ss[1:]], dtype=numpy.float32)
        
s = numpy.sqrt((w * w).sum(1))
w /= s.reshape((s.shape[0], 1))         # normalize w

try:
    while True:
        q = six.moves.input('>> ')      # input word = q
        if q not in word2index:
            print('"{0}" is not found'.format(q))
            continue
        v = w[word2index[q]]            # 入力単語の規格化された分散表現ベクトル
        similarity = w.dot(v)           # 入力単語と登録単語の類似度=行列の積
        print('query: {}'.format(q))
        
        count = 0
        for i in (-similarity).argsort():
            if numpy.isnan(similarity[i]):  # zeroは飛ばす
                continue
            if index2word[i] == q:          # 単語そのものは飛ばす
                continue
            print('{0}: {1}'.format(index2word[i], similarity[i]))
            count += 1
            if count == n_result:
                break

except EOFError:
    pass

