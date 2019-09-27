import pandas as pd
import json
from math import log
import os
from tqdm import tqdm


class HMM(object):
    def __init__(self, state_list=None):
        self.state_list = state_list or list('bmes')
        self.pi_dic = {k: 0 for k in self.state_list}  # {'b': start_b}, start prob of pi('b')
        self.a_dic = {k0: {k1: 0 for k1 in self.state_list}
                      for k0 in self.state_list}  # {'b':{'m':trans_b}}, trans prob of a('m'|'b')
        self.b_dic = {k: {} for k in self.state_list}  # {'b': {word: emit_b}}, emit prob of b(word|'b')
        self.count_dic = {k: 0 for k in self.state_list}  # {'b': num} num of 'b' in all state
        self.char2state = None
        self.default = -50  # default prob

    def base_cut(self, fpath=None, char_list=None, state_list=None):
        """input data is entire sentences, we can count all trans probs according to inputs
        char_list: sentence which is cut into chars, eg: [[char_1, char_2...], ..., [char_n1, char_n2...]]
        state: state of each sen each char, eg: [[b, m, ...], ..., [s, s, ...]]
        """

        if char_list is None or state_list is None:
            char_list, state_list = self.get_item(fpath)

        line_num = len(char_list)

        # count the prob of states
        for i in tqdm(range(line_num)):
            state = state_list[i]
            char = char_list[i]
            for j in range(len(state)):
                if j == 0:
                    self.pi_dic[state[j]] += 1
                    self.count_dic[state[j]] += 1
                else:
                    self.a_dic[state[j - 1]][state[j]] += 1
                    self.count_dic[state[j]] += 1
                    self.b_dic[state[j]][char[j]] = self.b_dic[state[j]].get(char[j], 0) + 1

        # calculate the prob
        # a_dic or b_dic equals 0 -> impossible to happen
        # log(x) to prevent overflow -> like, x*y can change to log(x) + log(y)
        for k in self.pi_dic.keys():
            if self.pi_dic[k] != 0:
                self.pi_dic[k] = log(self.pi_dic[k] / line_num)

        for k0 in self.a_dic.keys():
            for k1 in self.a_dic[k0].keys():
                if self.a_dic[k0][k1] != 0:
                    self.a_dic[k0][k1] = log(self.a_dic[k0][k1] / self.count_dic[k1])

        for k0 in self.b_dic.keys():
            for k1 in self.b_dic[k0].keys():
                if self.b_dic[k0][k1] != 0:
                    self.b_dic[k0][k1] = log(self.b_dic[k0][k1] / self.count_dic[k0])

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
                self.count_dic['s'] += fre
                self.b_dic['s'][word[0]] = self.b_dic['s'].get(word[0], 0) + fre
            else:
                self.count_dic['b'] += fre
                self.count_dic['e'] += fre
                self.b_dic['b'][word[0]] = self.b_dic['b'].get(word[0], 0) + fre
                self.b_dic['e'][word[-1]] = self.b_dic['e'].get(word[-1], 0) + fre
                for i in word[1:-1]:
                    self.b_dic['m'][i] = self.b_dic['m'].get(i, 0) + fre
                    self.count_dic['m'] += fre
                if len(word) == 2:
                    self.a_dic['b']['e'] += fre
                else:
                    self.a_dic['b']['m'] += fre
                    self.a_dic['m']['e'] += fre
                    self.a_dic['m']['m'] += fre * (len(word) - 3)

        total = sum(self.count_dic.values())
        for i in list('bs'):
            self.pi_dic[i] = log(self.count_dic[i] / total)

        for k0 in self.b_dic.keys():
            for k1 in self.b_dic[k0].keys():
                self.b_dic[k0][k1] = log(self.b_dic[k0][k1] / self.count_dic[k0])

        for k0 in self.a_dic.keys():
            for k1 in self.a_dic[k0].keys():
                if self.a_dic[k0][k1] != 0:
                    self.a_dic[k0][k1] = log(self.a_dic[k0][k1] / self.count_dic[k0])

        for i in list('se'):
            for j in list('sb'):
                self.a_dic[i][j] = self.pi_dic[j]

    def get_char2state(self):
        # state:word -> word:state
        default = {k: self.default for k in self.state_list}
        self.char2state = {}
        for i, j in self.b_dic.items():
            for k in j:
                if k not in self.char2state:
                    self.char2state[k] = default.copy()
                self.char2state[k][i] = j[k]

    def get_item(self, fpath):
        char_list = []
        state = []

        for line in open(fpath, 'r', encoding='utf-8'):
            line = line.replace('\n', '')
            if line == '':
                continue
            state.append([])
            char_list.append([])
            for i in line.split(' '):
                if i == '':
                    continue
                char_list[-1] += i
                state[-1].extend(self.get_default_state(i))
        return char_list, state

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

        if self.char2state is None:
            self.get_char2state()

        default = {k: self.default for k in self.state_list}
        nodes = [self.char2state.get(i, default) for i in text]
        nodes[0] = {i: j for i, j in nodes[0].items() if i in start_state}
        nodes[-1] = {i: j for i, j in nodes[-1].items() if i in end_state}

        paths = {n: ns + self.pi_dic[n] for n, ns in nodes[0].items()}

        for i in range(1, len(nodes)):
            paths_old, paths = paths, {}
            for n, ns in nodes[i].items():
                max_path, max_score = [], -1e10
                for p, ps in paths_old.items():
                    if self.a_dic[p[-1]][n] != 0:
                        score = ns + ps + self.a_dic[p[-1]][n]
                        if score > max_score:
                            max_path, max_score = p + n, score
                paths[max_path] = max_score

        key, value = '', -1e10
        for k, v in paths.items():
            if v > value:
                key, value = k, v

        return [i for i in key]

    def cut(self, text, start_state=None, end_state=None):
        key = self.viterbi(text, start_state, end_state)
        cut_text = []
        for i, s in enumerate(key):
            if s in start_state:
                cut_text.append('')
            cut_text[-1] += text[i]
        return cut_text

    def load(self):
        with open('saver/a_dic.json', 'r', encoding='utf-8') as f:
            self.a_dic = json.load(f)
        with open('saver/b_dic.json', 'r', encoding='utf-8') as f:
            self.b_dic = json.load(f)
        with open('saver/pi_dic.json', 'r', encoding='utf-8') as f:
            self.pi_dic = json.load(f)
        self.get_char2state()

    def save(self):
        if not os.path.exists('saver'):
            os.mkdir('saver')
        with open('saver/a_dic.json', 'w', encoding='utf-8') as f:
            json.dump(self.a_dic, f)
        with open('saver/b_dic.json', 'w', encoding='utf-8') as f:
            json.dump(self.b_dic, f)
        with open('saver/pi_dic.json', 'w', encoding='utf-8') as f:
            json.dump(self.pi_dic, f)


if __name__ == '__main__':
    hmm = HMM(list('bmes'))

    # fp = 'datagrand/words.txt'
    # hmm.base_cut(fp)

    # fp = 'dic.txt'
    # hmm.base_dict(fp)

    # hmm.save()

    hmm.load()

    text = '结婚和尚未结婚的人'
    ret = hmm.viterbi(text)
    print(ret)
