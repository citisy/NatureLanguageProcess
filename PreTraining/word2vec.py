import collections
import numpy as np
from utils import *
from tqdm import tqdm
import json

ns = 0
hs = 1
sg = 0
CBOW = 1


class Word2vec(object):
    def __init__(self, sentences=None, window=5, min_reduce=5, max_reduce=0.005,
                 layer1_size=100, table_size=1e6, alpha=0.025, negative=5,
                 model_mode=hs, train_mode=CBOW, itera=5):
        """
        :param sentences: 输入单词序列，iterable of iterables
        :param window: n-gram的窗口值
        :param min_reduce: 舍弃词频小于指定值的单词
        :param layer1_size: hs的大小
        :param table_size: neg的大小
        :param alpha: 学习率
        :param negative: 负采样样本数
        :param model_mode: 1 -> CBOW, 0 -> sg
        :param train_mode: 1 -> hs, 0 -> ns
        :param itera: 迭代次数
        """
        self.window = window
        self.layer1_size = layer1_size
        self.table_size = int(table_size)
        self.alpha = alpha
        self.negative = negative
        self.model_mode = model_mode
        self.train_mode = train_mode
        self.itera = itera
        self.sentence = sentences
        self.min_reduce = min_reduce
        self.max_reduce = max_reduce

    def train(self):
        # 按词频排序，制作字典
        word_list = []
        for line in self.sentence:
            word_list += line
        self.word_index = collections.Counter(word_list)

        # 过滤低频词
        self.word_index = {k: v for k, v in self.word_index.items() if v >= self.min_reduce}

        # 过滤高频词
        word_index = dict()
        d = len(word_list)
        for k, v in self.word_index.items():
            fre = v / d
            if fre >= self.max_reduce:
                prob = 1 - np.sqrt(self.max_reduce / fre)
                ran = np.random.random()
                if prob > ran:
                    word_index[k] = v
            else:
                word_index[k] = v

        self.word_index = word_index

        self.sigmoid_table = 1.0 / (1.0 + np.exp(-np.arange(-6, 6, 0.0001)))  # 快速sigmoid表

        if self.train_mode:
            self.create_binary_tree()
        else:
            self.init_unigram_table()

        self.index_word = {v: k for k, v in self.word_index.items()}

        # 将单词序列转为索引序列
        self.sentence_index = [[self.word_index.get(word, -1) for word in line] for line in self.sentence]

        self.vocab_size = len(self.word_index)

        # 随机初始化词向量，矩阵大小 => [vocab_size, layer1_size]
        self.h = np.random.random((self.vocab_size, self.layer1_size)) * 2 - 1

        # 初始化辅助权重矩阵
        # 在hs中表示每个节点的权值，在ns中表示隐藏层到输出层的权值
        self.v = np.zeros((self.vocab_size, self.layer1_size))

        for i in range(self.itera):
            if self.model_mode:
                self.CBOW()
            else:
                self.skip_gram()

    def fast_sigmoid(self, x):
        """通过查表近似计算sigmoid函数"""
        if x < -6:
            return 0
        if x > 6:
            return 1
        return self.sigmoid_table[int((x + 6) / 0.0001)]

    @count_time(output='CreateBinaryTree successful!')
    def create_binary_tree(self):
        """建立哈夫曼树"""
        # 字典从大到小排序，以便创建哈夫曼树
        word_count = sorted(self.word_index.items(), key=lambda x: x[1], reverse=True)
        word_index = dict()  # 按照词频编码
        cn = []

        for k, v in word_count:
            word_index[k] = len(word_index)
            cn.append(v)

        vocab_size = len(word_index)

        # 哈夫曼树用数组形式表示，[:vocab_size]保存单词，[vocab_size:]保存父节点
        # 作者源码中，矩阵大小为 2*vocab_size+1，但后续操作中，矩阵的最后2位貌似并未被使用，暂未知为什么要多出2位
        count = np.zeros(2 * vocab_size - 1, dtype=int)  # 保存结点的值
        count[:vocab_size] = cn
        count[vocab_size:] = 1e15

        parent_node = ['' for _ in range(2 * vocab_size - 2)]  # 保存父节点的下标
        binary = [0 for _ in range(2 * vocab_size - 1)]  # 保存编码

        # 从vocab_size开始查找，因为已经按词频排序，所以从中间向两边比较即可创建哈夫曼树
        pos1 = vocab_size - 1
        pos2 = vocab_size

        for a in range(vocab_size - 1):  # 哈夫曼树的总结点数为2*vocab_size-1
            # 最小值
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min1 = pos1
                    pos1 -= 1
                else:
                    min1 = pos2
                    pos2 += 1
            else:
                min1 = pos2
                pos2 += 1

            # 次最小值
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min2 = pos1
                    pos1 -= 1
                else:
                    min2 = pos2
                    pos2 += 1
            else:
                min2 = pos2
                pos2 += 1

            # 从vocab_size开始依次保存父节点的值
            count[vocab_size + a] = count[min1] + count[min2]
            parent_node[min1] = vocab_size + a
            parent_node[min2] = vocab_size + a
            # 次最小值即右孩子赋1
            binary[min2] = 1

        # 哈夫曼编码
        self.codelen = [0 for _ in range(vocab_size)]  # 保存编码长度
        self.code = [[] for _ in range(vocab_size)]  # 保存哈夫曼编码
        self.point = [[] for _ in range(vocab_size)]  # 保存节点序号

        for a in range(vocab_size):
            b = a
            i = 0
            while b < vocab_size * 2 - 2:
                self.code[a].append(binary[b])
                self.point[a].append(b - vocab_size)
                i += 1
                b = parent_node[b]

            self.codelen[a] = i
            self.code[a] = self.code[a][::-1]
            self.point[a] = self.point[a][::-1]

        # 返回传值
        self.word_index = word_index

    @count_time(output='InitUnigramTable successful!')
    def init_unigram_table(self):
        # 按照词频编码
        word_count = sorted(self.word_index.items(), key=lambda x: x[1], reverse=True)
        word_index = dict()
        cn = []

        for k, v in word_count:
            word_index[k] = len(word_index)
            cn.append(v)

        vocab_size = len(word_index)
        power = 0.75
        table = np.zeros(self.table_size, dtype=int)
        train_words_pow = 0

        for a in range(vocab_size):
            train_words_pow += np.power(cn[a], power)

        i = 0  # 单词下标
        d1 = np.power(cn[i], power) / train_words_pow

        for a in range(self.table_size):
            table[a] = i

            # 把table按词频划分，词频越高，占table的位置越多
            if a / self.table_size > d1:
                i += 1
                d1 += np.power(cn[i], power) / train_words_pow

            if i >= vocab_size:
                i = vocab_size - 1

        # 返回传值
        self.word_index = word_index
        self.table = table

    def CBOW(self):
        """词袋模型，上下文预测当前单词"""
        for aa, line in enumerate(tqdm(self.sentence_index)):
            if aa % 10000 == 0:
                self.alpha *= (1 - aa / len(self.sentence_index))

                if self.alpha <= 10e-4:
                    self.alpha = 10e-4

            for a, i in enumerate(line):
                # 低频词不训练
                if i == -1:
                    continue

                neu1e = np.zeros(self.layer1_size)

                # in layer -> hidden layer
                # 对指定单词随机前后共window个单词的权值进行更新，平均池化
                random_left = np.random.randint(self.window)
                l = a - random_left if a > random_left else 0
                r = a - random_left + self.window
                r = r if r < len(line) else len(line)

                idx = []
                for b in range(l, r):
                    if a == b:
                        continue
                    if line[b] == -1:
                        continue

                    idx.append(line[b])

                if not idx:
                    continue

                x = np.sum(self.h[idx], axis=0) / len(idx)

                # 训练自身隐藏层结点权值、自身词向量更新系数
                if self.train_mode:  # HS
                    for d in range(self.codelen[i]):
                        point = self.point[i][d]  # 路径上的点的序号

                        if point < 0:  # 小于0为叶结点，即单词自身，不迭代
                            continue

                        f = self.fast_sigmoid(x @ self.v[point].T)

                        g = (1 - self.code[i][d] - f) * self.alpha

                        # 记录累积误差项
                        neu1e += g * self.v[point]

                        # 更新非叶结点权重
                        self.v[point] += g * x

                else:  # NS
                    # 随机采个数最多为negative的负样本
                    for d in range(self.negative + 1):
                        # 第一个采样该单词，为正样本，其余采样为负样本
                        if d == 0:
                            collection_word_index = i
                            label = 1
                        else:
                            rand = np.random.randint(self.table_size)
                            collection_word_index = self.table[rand]

                            # 若采样落在该单词占有的区域，则跳过
                            # 词频越高，跳过几率越大，最终采到的样本越少
                            if collection_word_index == i:
                                continue

                            label = 0

                        # 计算f
                        f = self.fast_sigmoid(x @ self.v[collection_word_index].T)

                        g = (label - f) * self.alpha  # 计算学习率

                        # 记录累积误差项
                        neu1e += g * self.v[collection_word_index]

                        # 更新负样本权重
                        self.v[collection_word_index] += g * x

                # hidden -> in
                # 更新词向量，把误差向量加到上下文的向量上
                self.h[idx] += neu1e / len(idx)


    def skip_gram(self):
        for aa, line in enumerate(tqdm(self.sentence_index)):
            if aa % 10000 == 0:
                self.alpha *= (1 - aa / len(self.sentence_index))

                if self.alpha <= 10e-4:
                    self.alpha = 10e-4

            for a, i in enumerate(line):
                x = self.h[i]
                neu1e = np.zeros(self.layer1_size)

                random_left = np.random.randint(self.window)
                l = a - random_left if a > random_left else 0
                r = a - random_left + self.window
                r = r if r < len(line) else len(line)

                for b in range(l, r):
                    if a == b:
                        continue

                    if self.train_mode:  # HS
                        for d in range(self.codelen[i]):
                            point = self.point[i][d]
                            if point < 0:
                                continue

                            # hidden -> out
                            f = self.h[line[b]] @ self.v[point].T
                            f = self.fast_sigmoid(f)

                            g = (1 - self.code[i][d] - f) * self.alpha

                            # 记录累积误差项
                            neu1e += g * self.v[point]

                            # 更新非叶结点权重
                            self.v[point] += g * x

                    else:  # NS
                        for d in range(self.negative + 1):
                            # 第一个采样该单词，为正样本，其余采样为负样本
                            if d == 0:
                                collection_word_index = i
                                label = 1
                            else:
                                rand = np.random.randint(self.table_size)
                                collection_word_index = self.table[rand]

                                # 若采样落在该单词占有的区域，则跳过
                                # 词频越高，跳过几率越大，最终采到的样本越少
                                if collection_word_index == i:
                                    continue

                                label = 0

                            # hidden -> out
                            f = np.sum(self.h[line[b]] * self.v[collection_word_index])
                            f = self.fast_sigmoid(f)

                            # 计算下降梯度
                            g = (label - f) * self.alpha

                            # 记录累积误差项
                            neu1e += g * self.v[collection_word_index]

                            self.v[collection_word_index] += g * x

                    # in -> hidden
                    # 误差向量加到当前词的词向量上
                    self.h[i] += neu1e

    def save(self, save_path):
        np.save(save_path, self.h)

        with open(save_path + '.json', 'w', encoding='utf8') as f:
            json.dump(self.word_index, f, indent=4, ensure_ascii=False)

    def load(self, save_path):
        self.h = np.load(save_path + '.npy')

        with open(save_path + '.json', 'r', encoding='utf8') as f:
            self.word_index = json.load(f)

        self.index_word = {v: k for k, v in self.word_index.items()}

        self.vocab_size = len(self.word_index)

    def most_similar(self, s, cn=10):
        """余弦相似度判断相近词"""
        if s not in self.word_index:
            raise ValueError(f'The word {s} not in dict!')

        idx = self.word_index[s]
        sim = np.zeros(self.vocab_size)

        a = self.h[idx]
        for i in range(self.vocab_size):
            b = self.h[i]
            sim[i] = a @ b.T / (np.linalg.norm(a) * np.linalg.norm(b))

        arg_sort = np.argsort(sim)[::-1]

        return [(self.index_word[arg_sort[i + 1]], sim[arg_sort[i + 1]]) for i in range(cn)]


def my_model():
    save_path = 'saver/1998_corpus_my_word2vec_cbow_hs.model'

    fpath = '../data/1998_corpus_cut.txt'
    char_list = read_word_cut_file(fpath)
    model = Word2vec(char_list, model_mode=CBOW, train_mode=ns, itera=1)
    model.train()
    model.save(save_path)

    # model = Word2vec()
    # model.load(save_path)

    print(model.most_similar('计算机', 10))


def gensim_model():
    import gensim

    class MyCallback(gensim.models.callbacks.CallbackAny2Vec):
        epoch = 1

        def on_epoch_end(self, model):
            print(f'The {self.epoch}th epoch end!')
            self.epoch += 1

    save_path = 'saver/2014_corpus_word2vec_cbow_hs.model'
    # fpath = '../data/2014_corpus_cut.txt'
    # char_list = read_word_cut_file(fpath)
    #
    # model = gensim.models.Word2Vec(char_list, min_count=5, size=200, sg=0, hs=1,
    #                                callbacks=[MyCallback()])
    # model.callbacks = ()  # 需要清空回调函数，否则保存会报错！
    # model.save(save_path)

    model = gensim.models.Word2Vec.load(save_path)

    print(model.wv.most_similar('计算机'))


if __name__ == '__main__':
    my_model()
    # gensim_model()
