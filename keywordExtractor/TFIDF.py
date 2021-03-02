import math
import numpy as np


class TFIDF:
    def get_tf(self, cut_corpus, word_len=2):
        """每一行为一个文档"""
        word_tf = []
        for line in cut_corpus:
            tf = {}
            n_word = len(line)  # 统计每一个文档的词的数量
            for word in line:
                if word >= word_len:
                    tf[word] = tf.get(word, 0.) + 1. / n_word  # 统计每一个文档中每一个出现的次数

            word_tf.append(tf)

        return word_tf

    def get_idf(self, cut_corpus, word_len=2):
        word_idf = {}
        num_doc = len(cut_corpus)  # 文档总数，这里每一行为一个文档

        for line in cut_corpus:
            for word in set(line):
                if len(word) >= word_len:
                    word_idf[word] = word_idf.get(word, 0) + 1  # 统计每个文档某个词是否出现

        for k, v in word_idf.items():
            word_idf[k] = math.log((num_doc / (1. + v)))

        return word_idf

    def get_tf_vec(self, cut_corpus, word_len=2):
        """整个输入当作一个文档"""
        word_tf_vec = {}
        for line in cut_corpus:
            for word in line:
                if len(word) >= word_len:  # 单词长度小于阈值丢弃
                    word_tf_vec[word] = word_tf_vec.get(word, 0.) + 1.

        num_word = sum(word_tf_vec.values())
        for k, v in word_tf_vec.items():
            word_tf_vec[k] = v / num_word

        return word_tf_vec

    def load_idf(self, idf_path):
        word_idf = {}
        for line in open(idf_path, 'r', encoding='utf8'):
            word, idf = line.replace('\n', '').split(' ')
            word_idf[word] = float(idf)

        # 缺省的单词取中值
        default_idf = sorted(word_idf.values())[len(word_idf) // 2]

        return word_idf, default_idf

    def get_feature_list(self, cut_corpus, word_len=2):
        word_tf = self.get_tf(cut_corpus, word_len)
        word_idf = self.get_idf(cut_corpus, word_len)
        feature_list = []
        for tf in word_tf:
            f_dict = {}
            for k, v in tf.items():
                f_dict[k] = v * word_idf[k]

            feature_list.append(f_dict)

        return feature_list

    def get_feature_matrix(self, cut_corpus, word_len=2):
        word_tf = self.get_tf(cut_corpus, word_len)
        word_idf = self.get_idf(cut_corpus, word_len)
        word_list = list(word_idf.keys())
        word_size = len(word_list)
        num_sentence = len(word_tf)
        feature_matrix = np.zeros((num_sentence, word_size))

        for i, tf in enumerate(word_tf):
            for k, v in tf.items():
                feature_matrix[i][word_list.index(k)] = v * word_idf[k]

        return feature_matrix

    def extract_keyword(self, cut_corpus, topK=10, word_len=2):
        """短文本关键词抽取，把整个输入看作是一个文档。"""
        word_tf_vec = self.get_tf_vec(cut_corpus, word_len)
        word_idf, default_idf = self.load_idf('../data/idf.txt')
        word_tfidf = {}

        for k in word_tf_vec:
            word_tfidf[k] = word_tf_vec[k] * word_idf.get(k, default_idf)

        sort = sorted(word_tfidf.items(), key=lambda x: x[1], reverse=True)

        return sort[:topK]


def sklearn_tfidf():
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer

    with open('../data/peoples_daily_word_cut.txt', 'r', encoding='utf8') as f:
        cut_corpus = [line for line in f.read().split('\n')]

    vec = CountVectorizer()
    x = vec.fit_transform(cut_corpus)

    tfidf = TfidfTransformer()
    feature = tfidf.fit_transform(x)

    print(feature.toarray())


def jieba_tfidf_extract_keyword():
    from jieba.analyse import extract_tags

    with open('../data/peoples_daily_word_cut.txt', 'r', encoding='utf8') as f:
        original_corpus = f.read().replace(' ', '')

    keywords = extract_tags(original_corpus, topK=10)
    print(keywords)


if __name__ == '__main__':
    model = TFIDF()

    with open('../data/peoples_daily_word_cut.txt', 'r', encoding='utf8') as f:
        cut_corpus = [line.split(' ') for line in f.read().split('\n')]

    # feature_list = model.get_feature_list(cut_corpus)
    # print(feature_list)

    # sklearn_tfidf()

    keywords = model.extract_keyword(cut_corpus)
    print(keywords)

    jieba_tfidf_extract_keyword()
