import re


def read_word_cut_file(fp, line_break='\n', word_break=' '):
    """读取标准格式的分词文本

    输入文本格式: word1 word2 word3...

    params:
        word_break: 单词分割符
        line_break: 段落分割符
    """
    with open(fp, 'r', encoding='utf8') as f:
        cut_corpus = [line.split(word_break) for line in f.read().split(line_break)]

    return cut_corpus


def read_word_tag_file(fp, line_break='\n', word_break=' '):
    """读取标准格式的已打标的分词文本

    输入文本格式:
        char1 tag1
        char2 tag2
        char3 tag3
        ...

    params:
        word_break: 单词分割符
        line_break: 段落分割符
    """
    word_list, tag_list = [], []

    with open(fp, 'r', encoding='utf8') as f:
        for day in f.read().split(line_break * 5):
            for corpus in day.split(line_break * 4):
                for line in corpus.split(line_break * 3):
                    for sentence in line.split(line_break * 2):
                        if len(sentence) == 0:
                            continue
                        word_list.append([])
                        tag_list.append([])
                        for word_tag in sentence.split(line_break):
                            word, tag = word_tag.split(word_break)
                            word_list[-1].append(word)
                            tag_list[-1].append(tag)

    return word_list, tag_list


def make_word_cut_file(input_file, output_file,
                       line_break='\n', word_break=' ', tag_break='/'):
    """将输入打标文本转换为标准格式的分词文本

    输入文本格式:
        word1/tag1 word2/tag2...

    输出文本格式:
        word1 word2...

    params:
        word_break: 单词分割符
        line_break: 段落分割符
        tag_break: 标签分割符

    """
    with open(input_file, 'r', encoding='utf8') as fi:
        with open(output_file, 'w', encoding='utf8') as fo:
            s = fi.read()
            s = re.sub('\[', '', s)
            s = re.sub('\]/[a-z]+', '', s)
            for line in s.split(line_break):
                for word_tag in line.strip().split(word_break):
                    fo.write(word_tag.split(tag_break)[0] + word_break)
                fo.write(line_break)


def make_4_tags_file(input_file, output_file, line_break='\n', word_break=' ',
                     tags=tuple('bmes')):
    """将输入打标文本转换为标准格式的4标签分词文本

    输入文本格式:
        word1 word2...

    输出文本格式:
        char1 tag1
        char2 tag2
        ...

    params:
        word_break: 单词分割符
        line_break: 段落分割符
        tag_break: 标签分割符

    """

    def get_tag(word):
        """将中文单词转换为4标签

        eg:
            我爱你 -> [b, m, s]
        """
        output_tag = []
        if len(word) == 1:
            output_tag.append(tags[3])
        elif len(word) == 2:
            output_tag = [tags[0], tags[2]]
        else:
            m_num = len(word) - 2
            m_list = [tags[1]] * m_num
            output_tag.append(tags[0])
            output_tag.extend(m_list)  # 把M_list中的'M'分别添加进去
            output_tag.append(tags[2])

        return output_tag

    with open(input_file, 'r', encoding='utf8') as fi:
        with open(output_file, 'w', encoding='utf8') as fo:
            for line in fi.read().split(line_break):
                for word in line.strip().split(word_break):
                    tag = get_tag(word)
                    for i in range(len(word)):
                        fo.write(f'{word[i]} {tag[i]}' + line_break)

                fo.write(line_break)


if __name__ == '__main__':
    # make_word_cut_file('data/2014_corpus.txt', 'data/2014_corpus_cut.txt')

    make_4_tags_file('data/2014_corpus_cut.txt', 'data/2014_corpus_4_tags.txt')

    # word_list, tag_list = read_word_tag_file('data/peoples_daily_4_tag.txt')

    # word = read_word_cut_file('data/peoples_daily_word_cut.txt')
