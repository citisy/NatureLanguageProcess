import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import utils


class HMM(object):
    def __init__(self, state_list=None):
        self.state_list = state_list or list('bmes')

        # eg: {'b': start_b}, start prob of pi('b')
        self.pi = {k: 0 for k in self.state_list}

        # eg: {'b':{'m':trans_b}}, trans prob of a('m'|'b')
        self.a = {k0: {k1: 0 for k1 in self.state_list}
                  for k0 in self.state_list}

        # eg: {'b': {word: emit_b}}, emit prob of b(word|'b')
        self.b = {k: {} for k in self.state_list}

        # eg: {'b': num} num of 'b' in all state
        self.count = {k: 0 for k in self.state_list}

        self.char2state = None
        self.default = -1e9  # 不可能存在的情况的默认对数概率值

    def base_cut(self, fpath=None, char_list=None, state_list=None):
        """input data is entire sentences, we can count all trans probs according to inputs
        char_list: sentence which is cut into chars, eg: [[char_1, char_2...], ..., [char_n1, char_n2...]]
        state: state of each sen each char, eg: [[b, m, ...], ..., [s, s, ...]]
        """

        if char_list is None or state_list is None:
            char_list, state_list = utils.read_word_tag_file(fpath)

        line_num = len(char_list)

        # count the prob of states
        for i in tqdm(range(line_num)):
            state = state_list[i]
            char = char_list[i]
            for j in range(len(state)):
                self.count[state[j]] += 1

                if j == 0:
                    self.pi[state[j]] += 1
                else:
                    self.a[state[j - 1]][state[j]] += 1
                    self.b[state[j]][char[j]] = self.b[state[j]].get(char[j], 0) + 1

        # calculate the prob
        # a_dic or b_dic equals 0 -> impossible to happen
        # log(x) to prevent overflow -> like, x*y can change to log(x) + log(y)
        for k in self.pi.keys():
            if self.pi[k] != 0:
                self.pi[k] = np.log(self.pi[k] / line_num)
            else:
                self.pi[k] = self.default

        for k0 in self.a.keys():
            for k1 in self.a[k0].keys():
                if self.a[k0][k1] != 0:
                    self.a[k0][k1] = np.log(self.a[k0][k1] / self.count[k1])
                else:
                    self.a[k0][k1] = self.default

        for k0 in self.b.keys():
            for k1 in self.b[k0].keys():
                if self.b[k0][k1] != 0:
                    self.b[k0][k1] = np.log(self.b[k0][k1] / self.count[k0])
                else:
                    self.b[k0][k1] = self.default

    def base_dict(self, fpath):
        """input data is a dict of words and counts, we can't get a whole trans probs according to inputs,
        so we need to assume some trans probs:
            we have no trans of 'ss', 'sb', 'es', 'eb',
            but in fact, we can assume that all sens are made up of random words, and each contexts are independence,
            so the trans of each tags are also independence.
            we can get this approximate calculation:
                a(s|s) = pi(s), a(b|s) = pi(b), ...
        ---------
        fmt of inputs:
            |word1 count1|
            |word2 counts|
            |     ...    |
            count: words' frequency
        """
        df = pd.read_csv(fpath, sep=' ')
        word_dic = {k: v for k, v in df.iloc[:, 0:2].values}
        for word, fre in word_dic.items():
            if len(word) == 1:
                self.count['s'] += fre
                self.b['s'][word[0]] = self.b['s'].get(word[0], 0) + fre
            else:
                self.count['b'] += fre
                self.count['e'] += fre
                self.b['b'][word[0]] = self.b['b'].get(word[0], 0) + fre
                self.b['e'][word[-1]] = self.b['e'].get(word[-1], 0) + fre

                for i in word[1:-1]:
                    self.b['m'][i] = self.b['m'].get(i, 0) + fre
                    self.count['m'] += fre

                if len(word) == 2:
                    self.a['b']['e'] += fre
                else:
                    self.a['b']['m'] += fre
                    self.a['m']['e'] += fre
                    self.a['m']['m'] += fre * (len(word) - 3)

        total = sum(self.count.values())

        for i in list('bs'):
            self.pi[i] = np.log(self.count[i] / total)

        for k0 in self.b.keys():
            for k1 in self.b[k0].keys():
                self.b[k0][k1] = np.log(self.b[k0][k1] / self.count[k0])

        for k0 in self.a.keys():
            for k1 in self.a[k0].keys():
                if self.a[k0][k1] != 0:
                    self.a[k0][k1] = np.log(self.a[k0][k1] / self.count[k0])
                else:
                    self.a[k0][k1] = self.default

        for i in list('se'):
            for j in list('sb'):
                self.a[i][j] = self.pi[j]

    def get_char2state(self):
        """
        input: b_dic = {state: word}
        output: char2state = {word: state}
        """
        default = {k: self.default for k in self.state_list}
        char2state = {}
        for i, j in self.b.items():
            for k in j:
                if k not in char2state:
                    char2state[k] = default.copy()
                char2state[k][i] = j[k]

        return char2state

    def get_default_state(self, input_str):
        """
        input_str: text, str type, eg: 我爱你
        output: states, list type, eg: ['b', 'm', 'e']
        """
        output = []
        if len(input_str) == 1:
            output.append('s')
        elif len(input_str) == 2:
            output = ['b', 'e']
        else:
            M_num = len(input_str) - 2
            M_list = ['m'] * M_num
            output.append('b')
            output.extend(M_list)  # 把M_list中的'M'分别添加进去
            output.append('e')

        return output

    def viterbi(self, text, start_state=None, end_state=None):
        start_state = start_state or list('bs')
        end_state = end_state or list('es')

        # 为了方便查询，对 b_dic 进行转置操作
        char2state = self.get_char2state()

        default = {k: self.default for k in self.state_list}

        # 后面会重复用到b_i的值，为了减少重复查询，这里预先查询保存
        nodes = [char2state.get(i, default) for i in text]
        nodes[0] = {i: j for i, j in nodes[0].items() if i in start_state}
        nodes[-1] = {i: j for i, j in nodes[-1].items() if i in end_state}

        paths_score = {tuple([n]): ns + self.pi[n] for n, ns in nodes[0].items()}

        for i in range(1, len(nodes)):
            paths_score_old, paths_score = paths_score, {}

            for o_t, b_i in nodes[i].items():
                # 默认值放大100倍，是为了防止未登录词造成的错误
                max_path, max_score = tuple(), self.default * 100

                for p, score_old in paths_score_old.items():
                    aji = self.a[p[-1]].get(o_t, self.default)
                    score = score_old + aji + b_i
                    if score > max_score:
                        max_path, max_score = p + tuple([o_t]), score

                if max_path:
                    paths_score[max_path] = max_score

        return max(paths_score.items(), key=lambda x: x[1])[0]

    def cut(self, text, start_state=None, end_state=None):
        start_state = start_state or list('bs')
        end_state = end_state or list('es')

        key = self.viterbi(text, start_state, end_state)
        cut_text = []
        for i, s in enumerate(key):
            if s in start_state:
                cut_text.append('')
            cut_text[-1] += text[i]

        return cut_text

    def load(self):
        self.a = pd.read_table('saver/a.tsv', index_col=0).to_dict()
        self.b = pd.read_table('saver/b.tsv', index_col=0).to_dict()
        self.pi = pd.read_table('saver/pi.tsv', index_col=0, header=None)[1].to_dict()

    def save(self):
        if not os.path.exists('saver'):
            os.mkdir('saver')

        pd.DataFrame(self.a).to_csv('saver/a.tsv', sep='\t')
        pd.DataFrame(self.b).to_csv('saver/b.tsv', sep='\t')
        pd.Series(self.pi).to_csv('saver/pi.tsv', sep='\t')


if __name__ == '__main__':
    hmm = HMM()

    # fp = '../data/2014_corpus_4_tags.txt'
    # hmm.base_cut(fp)

    # fp = '../data/dic.txt'
    # hmm.base_dict(fp)

    # hmm.save()

    hmm.load()

    print(hmm.cut('结婚和尚未结婚的人'))
    print(hmm.cut('隐马尔可夫模型中有两个序列，一个是状态序列，另一个是观测序列，其中状态序列是隐藏的。'))

"""
['结婚', '和', '尚未', '结婚', '的', '人']
['隐马尔', '可夫', '模型', '中有', '两个', '序列', '，', '一个', '是', '状态', '序列', '，', '另一个', '是', '观测', '序列', '，', '其中', '状态', '序列', '是', '隐藏', '的', '。']
"""
