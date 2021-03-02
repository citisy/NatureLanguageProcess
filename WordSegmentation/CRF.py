"""refer to: https://github.com/lancifollia/crf"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import utils
from scipy.optimize import fmin_l_bfgs_b


class LinearCRF:
    def __init__(self, states=None):
        self.states = states or list('bmes')
        self.states_dict = {state: i + 1 for i, state in enumerate(self.states)}

        # 定义初始状态的上一个转移状态prev_y
        self.states_dict['#'] = 0

        self.feature_dict = dict()
        self.feature_set = set()
        self.empirical_counts = dict()

    def train(self, fpath=None, char_list=None, state_list=None):
        if char_list is None or state_list is None:
            char_list, state_list = utils.read_word_tag_file(fpath)

        state_list = [[self.states_dict[b] for b in a] for a in state_list]

        self.build_features(char_list, state_list)

        w = np.zeros(len(self.feature_set))

        self.w, _, _ = fmin_l_bfgs_b(func=self.loss_func, fprime=self.fprime,
                                     x0=w, args=(char_list,))

    def loss_func(self, w, *args):
        char_list = args[0]

        e = np.zeros(len(self.feature_set))
        total_z = 0

        for x in tqdm(char_list):
            m = self.build_matrix(w, x)
            alpha, beta, z, scaling_dic = self.forward_backward(m, len(x))
            total_z += np.log(z) + sum(np.log(s) for s in scaling_dic.values())

            for t in range(len(x)):
                m_i = m[t]
                for labels in self.feature_dict[x[t]]:
                    # 计算状态概率
                    if labels[0] == -1:
                        prob = (alpha[t, labels[1]] * beta[t, labels[1]]) / z
                        if t in scaling_dic:
                            prob *= scaling_dic[t]

                    # 计算转移概率
                    elif t > 0:
                        prob = (alpha[t - 1, labels[0]] * m_i[labels] * beta[t, labels[1]]) / z
                    else:
                        prob = (m_i[0, labels[1]] * beta[t, labels[1]]) / z

                    fid = self.feature_dict[x[t]][labels]
                    e[fid] += prob

        likelihood = np.dot(self.empirical_counts, w) - total_z - np.sum(np.dot(w, w)) / (10 * 2)
        self.gradients = self.empirical_counts - e - w / 10

        return -likelihood

    def fprime(self, *args):
        return -self.gradients

    def build_features(self, char_list=None, state_list=None):
        """构建特征方程"""
        feature_dic = dict()
        empirical_counts = dict()
        feature_set = set()

        for i in range(len(char_list)):
            chars, states = char_list[i], state_list[i]
            for j in range(len(chars)):
                c, y = chars[j], states[j]
                if j > 0:
                    prev_y = states[j - 1]
                else:
                    prev_y = 0

                # 构建转移特征
                feature = (c, prev_y, y)
                if feature not in feature_set:
                    fid = len(feature_set)
                    feature_dic[c] = feature_dic.get(c, dict())
                    feature_dic[c][(prev_y, y)] = fid
                    feature_set.add(feature)
                else:
                    fid = feature_dic[c][(prev_y, y)]

                empirical_counts[fid] = empirical_counts.get(fid, 0) + 1

                # 构建状态特征
                feature = (c, y)
                if feature not in feature_set:
                    fid = len(feature_set)
                    feature_dic[c] = feature_dic.get(c, dict())
                    feature_dic[c][(-1, y)] = fid
                    feature_set.add(feature)
                else:
                    fid = feature_dic[c][(-1, y)]

                empirical_counts[fid] = empirical_counts.get(fid, 0) + 1

        self.feature_dict = feature_dic
        self.feature_set = feature_set
        self.empirical_counts = np.array([empirical_counts[fid] for fid in range(len(feature_set))])

    def build_matrix(self, w, x):
        """构建目标函数的矩阵表示形式"""
        m = list()
        n_states = len(self.states_dict)

        for t in range(len(x)):
            m_i = np.zeros((n_states, n_states))

            if x[t] in self.feature_dict:
                for labels in self.feature_dict[x[t]]:
                    w_i = w[self.feature_dict[x[t]][labels]]
                    if labels[0] == -1:
                        m_i[:, labels[1]] += w_i
                    else:
                        m_i[labels] += w_i

            m_i = np.exp(m_i)

            # 初始矩阵只保留 '#' 标志的权值
            if t == 0:
                m_i[1:] = 0
            # 非初始矩阵去除 '#' 标志的权值
            else:
                m_i[:, 0] = 0
                m_i[0, :] = 0

            m.append(m_i)

        return m

    def forward_backward(self, m, time_length, eps=1e20):
        """前向后向算法"""
        num_labels = len(self.states_dict)

        alpha = np.zeros((time_length, num_labels))
        alpha[0] = m[0][0, :]

        scaling_dic = dict()

        t = 1
        while t < time_length:
            overflow = False
            for label_id in range(1, num_labels):
                alpha[t, label_id] = np.dot(alpha[t - 1], m[t][:, label_id])

                if alpha[t, label_id] > eps:
                    overflow = True
                    scaling_dic[t - 1] = eps
                    break

            # 防止溢出，这一步而非必须，可以省略
            if overflow:
                alpha[t - 1] /= eps
                alpha[t] = 0
            else:
                t += 1

        beta = np.zeros((time_length, num_labels))
        beta[-1] = 1

        for t in range(time_length - 2, -1, -1):
            for label_id in range(1, num_labels):
                beta[t, label_id] = np.dot(beta[t + 1], m[t + 1][label_id])

            if t in scaling_dic.keys():
                beta[t] /= scaling_dic[t]

        z = np.sum(alpha[-1])

        return alpha, beta, z, scaling_dic

    def cut(self, text, start_state=None):
        start_state = start_state or list('bs')

        m = self.build_matrix(self.w, text)
        key = self.viterbi(text, m)
        key = [self.states[k - 1] for k in key]

        cut_text = []
        for i, s in enumerate(key):
            if s in start_state:
                cut_text.append('')
            cut_text[-1] += text[i]

        return cut_text

    def viterbi(self, text, m):
        n_states = len(self.states_dict)
        time_steps = len(text)
        path = np.zeros((time_steps, n_states), dtype=int)
        max_value_table = np.zeros((time_steps, n_states))

        max_value_table[0] = m[0][0]

        for t in range(1, time_steps):
            for state_id in range(1, n_states):
                max_value, max_value_id = -np.inf, -1

                for prev_state_id in range(1, n_states):
                    value = max_value_table[t - 1][prev_state_id] * m[t][prev_state_id, state_id]

                    if value > max_value:
                        max_value = value
                        max_value_id = prev_state_id

                max_value_table[t, state_id] = max_value
                path[t, state_id] = max_value_id

        max_path = np.argmax(max_value_table[-1])
        paths = [max_path]

        for t in range(time_steps - 1, 0, -1):
            max_path = path[t, max_path]
            paths.append(max_path)

        return paths[::-1]

    def save(self):
        data = dict()
        for char, feature_id in self.feature_dict.items():
            for feature, fid in feature_id.items():
                data[fid] = [char, *feature, self.w[fid]]

        df = pd.DataFrame(data).T
        df.columns = ['char', 'prev_y', 'y', 'w']
        df.to_csv('saver/feature.tsv', sep='\t')

    def load(self):
        df = pd.read_table('saver/feature.tsv', index_col=0, dtype={'char': str})
        self.w = np.zeros(len(df))

        for index, row in df.iterrows():
            self.feature_dict[row['char']] = self.feature_dict.get(row['char'], dict())
            self.feature_dict[row['char']][(row['prev_y'], row['y'])] = index
            self.w[index] = row['w']


if __name__ == '__main__':
    crf = LinearCRF()

    crf.train('../data/1998_corpus_4_tags.small.txt')
    # crf.save()
    # crf.load()

    print(crf.cut('全党和全国各族人民团结一致，坚持党的领导，坚持马列主义 、毛泽东思想，坚持人民民主专政 ，坚持社会主义道路。'))

"""['全党', '和', '全国', '各族', '人民', '团结', '一致', '，', '坚持', '党', '的', '领导', '，', '坚持', '马列主义 ', '、', '毛泽东思想', '，', '坚持', '人民', '民主', '专', '政 ', '，', '坚持', '社会主义', '道路', '。']"""
