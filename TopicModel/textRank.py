import numpy as np


class TextRank:
    def get_feature(self, cut_corpus, word_len, n_gram):
        word_set = set()
        for line in cut_corpus:
            for word in line:
                if len(word) >= word_len:
                    word_set.add(word)

        word_list = list(word_set)
        word_graph = np.zeros((len(word_set), len(word_set)), dtype=int)

        for line in cut_corpus:
            for i, present_word in enumerate(line):
                if len(present_word) >= word_len:
                    present_index = word_list.index(present_word)
                    for next_word in line[i + 1:i + 1 + n_gram]:
                        next_index = word_list.index(next_word)
                        word_graph[present_index, next_index] += 1
                        word_graph[next_index, present_index] += 1

        return word_list, word_graph

    def extract_keyword(self, cut_corpus, topK=10, word_len=2,
                        n_gram=2, d=0.85, iters=10, eps=1e-2):
        word_list, word_graph = self.get_feature(cut_corpus, word_len, n_gram)

        similarity = np.zeros(len(word_list)) + 1. / len(word_list)
        update_similarity = similarity.copy()

        for _ in range(iters):
            for i in word_list:
                s = 0
                for j in word_list:
                    s += word_graph[j, i] / word_graph[j].sum() * similarity[j]
                update_similarity[i] = 1 - d + d * s

            if np.abs(update_similarity - similarity).max() < eps:
                similarity = update_similarity
                break

            similarity = update_similarity

        argsort = np.argsort(similarity)[::-topK - 1]

        r = []
        for idx in argsort:
            r.append(word_list[idx])

        return r


def jieba_textRank_extract_keyword():
    from jieba.analyse import textrank

    with open('../data/peoples_daily_word_cut.txt', 'r', encoding='utf8') as f:
        original_corpus = f.read().replace(' ', '')

    keywords = textrank(original_corpus, topK=10)
    print(keywords)


if __name__ == '__main__':
    model = TextRank()

    with open('../data/peoples_daily_word_cut.txt', 'r', encoding='utf8') as f:
        cut_corpus = [line.split(' ') for line in f.read().split('\n')]

    model.extract_keyword(cut_corpus)
