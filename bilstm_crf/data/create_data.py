from sklearn.model_selection import train_test_split
import pickle


CORPUS_PATH = 'ChineseCorpus199801.txt'
SAVE_PATH = "preprocessed_corpus.pkl"

tag2id = {
    'B': 0,
    'M': 1,
    'E': 2,
    'S': 3,
}
id2tag = ['B', 'M', 'E', 'S']

id2word = []
word2id = {}


def get_list(input_str: str) -> list:
    """Transform single word to tag sequence.

    Args:
        input_str (str): Single word.
    
    Returns:
        list: The tag sequence of word `input_str`.
    """
      # tag sequence to be returned

    if len(input_str) == 1:  # "我"
        output_str = [tag2id['S']]
    elif len(input_str) == 2:  # "喜欢"
        output_str = [tag2id['B'], tag2id['E']]
    else:  # "太平洋"
        output_str = []
        M_num = len(input_str) - 2
        M_list = [tag2id['M']] * M_num
        output_str += [tag2id['B']]
        output_str += M_list
        output_str += [tag2id['E']]
    
    return output_str


def handle_data():
    """Preprocessed the corpus for Chinese word split."""
    x_data = []
    y_data = []
    word_num = 0  # 用于从字到索引值的映射
    line_num = 0
    
    print("=== INFO === Loading and processing corpus...")
    with open(CORPUS_PATH, 'r', encoding='gb2312', errors='ignore') as f:
        corpus = f.readlines()
    
    for line in corpus:  # 处理每一行句子  # "迈向 新 时代"
        line = line.split(' ')[1:]  # 删除空格，从字符序列变为词序列
        line = [l for l in line if len(l)]  # 删除句子中多余的空格
        line = [l.split('/')[0] for l in line]  # 删除每个词的语法标注
        line = [l.split('[')[1] if '[' in l else l for l in line]  # 删除每个词的语法标注
        line = ' '.join(line)  # 重新拼接为由空格分隔词的字符串

        line_num += 1  # 句子计数器自增
        line = line.strip()  # 裁剪空格和换行符
        if not line:  # 如果剪完还是空
            continue  # 则继续处理下一行
        line_x = []  # 用于存储这一行的训练数据，用字的索引值表示句子
        for i in range(len(line)):
            if line[i] == " ":  # 如果为空格
                continue  # 则跳到下一个字符
            if (line[i] in id2word):
                line_x.append(word2id[line[i]])
            else:
                id2word.append(line[i])
                word2id[line[i]] = word_num
                line_x.append(word_num)
                word_num = word_num+1
        
        x_data.append(line_x)

        lineArr = line.split(" ")  # 句子中的词列表
        line_y = []  # 用于存储这一个句子对应的 tag 序列
        for item in lineArr:
            line_y.extend(get_list(item))
        y_data.append(line_y)
    
    print(x_data[-3247])
    print([id2word[i] for i in x_data[-3247]])
    print(y_data[-3247])
    print([id2tag[i] for i in y_data[-3247]])
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=43)

    print("=== INFO === Saving preprocessed corpus...")
    with open(SAVE_PATH, 'wb') as outp:
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(tag2id, outp)
        pickle.dump(id2tag, outp)
        pickle.dump(x_train, outp)
        pickle.dump(y_train, outp)
        pickle.dump(x_test, outp)
        pickle.dump(y_test, outp)
    print("=== INFO === Finished...")


def test():
    print("我", get_list("我"))
    print("来自", get_list("来自"))
    print("俄罗斯", get_list("俄罗斯"))
    print("澳大利亚", get_list("澳大利亚"))


if __name__ == "__main__":
    handle_data()