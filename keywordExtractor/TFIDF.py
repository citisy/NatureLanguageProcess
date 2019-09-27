import math


def tfidf(cut_corpus):
    """tf 和 idf 的计算是完全分开的两部分"""
    word_tf = []
    word_idf = {}
    num_doc = len(cut_corpus)  # 文档总数，这里每一行为一个文档

    for line in cut_corpus:
        tf = {}
        n_word = len(line)  # 统计每一个文档的词的数量
        for word in line:
            tf[word] = tf.get(word, 0.) + 1. / n_word  # 统计每一个文档中每一个出现的次数

        word_tf.append(tf)

        for word in set(line):
            word_idf[word] = word_idf.get(word, 0) + 1  # 统计每个文档某个词是否出现

    for k, v in word_idf.items():
        word_idf[k] = math.log((num_doc / (1. + v)))

    feature_list = []
    for tf in word_tf:
        f_dict = {}
        for k, v in tf.items():
            f_dict[k] = v * word_idf[k]

        feature_list.append(f_dict)

    return feature_list


def sklearn_tfidf():
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer

    with open('../data/corpus_after_cut.txt', 'r', encoding='utf8') as f:
        cut_corpus = [line for line in f.read().split('\n')]

    vec = CountVectorizer()
    x = vec.fit_transform(cut_corpus)

    tfidf = TfidfTransformer()
    feature = tfidf.fit_transform(x)

    print(feature.toarray())


if __name__ == '__main__':
    with open('../data/corpus_after_cut.txt', 'r', encoding='utf8') as f:
        cut_corpus = [line.split(' ') for line in f.read().split('\n')]

    feature_list = tfidf(cut_corpus)
    print(feature_list)

    sklearn_tfidf()
