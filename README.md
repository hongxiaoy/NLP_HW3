1. 利用已经学习过的理论方法和北京大学标注的《人民日报》分词和词性标注语料，设计实现至少两种不同的汉语词语自动切分方法，进行性能测试和分析。然后利用不同类型的网络文本测试分词系统，

2. 对比分析分词方法和不同测试样本的性能变化。在得到的分词结果的基础上，实现子词压缩。

需要提交实现代码以及实验报告，不需要提交数据集。

Finished：P26-33

[TOC]

# 词典构建

代码位于 `data.py`：

```python
def load_vocab():

    # 读取语料库标注
    with open('ChineseCorpus199801.txt', 'r', encoding='gb2312', errors='ignore') as f:
        corpus = f.readlines()
    
    # # ['19980131-04-004-004/m', '这/r', '就/d', '是/v', '[江/j', '峡/j', '大道/n]ns', '。/w'] 64

    # 构建词典
    vocab = []
    for c in corpus:
        word_list = c.split(' ')  # 以空格分隔
        word_list = [w for w in word_list if len(w)]  # 去掉空字符串
        word_list = [w for w in word_list if w != '\n']  # 去掉行末换行符
        word_list = word_list[1:]  # 去掉开头日期
        word_list = [w.split('/')[0] for w in word_list]  # 去掉词性标注
        word_list = [w.split('[')[0] for w in word_list]  # 去掉特定地名前的大括号
        vocab.extend(word_list)  # 添加到词典
    vocab = list(set(vocab))  # 构建无重复词的词典
    
    return vocab
```

# 汉语分词方法

## 有词典切分

### 最大匹配法

#### 正向最大匹配算法

伪代码：（根据课件第八章第 30 页转化而来）

```python
def Partition(S, vocab):
    """
    Args:
        S (str): Input string with Chinese characters.
        vocab (list): The vocabulary that store the words.
    """
    results = []

    i = 0
    while True:
        n = len(S) - i
        if n == 1:
            return results
        m = get_max_len_of_words(vocab)
        if n < m:
            m = n
        
        w = S[i:i+m]
        while w not in vocab:
            if len(w) > 1:
                w = w[:-1]
            elif len(w) == 1:
                vocab.append(w)
                break
        results.append(w)
        i = i + len(w)
        if i == len(S):
            return results
```

其实上面的代码再加上一个从词表中获取最长单词的长度的函数就构成了正向最大匹配算法的代码。完整代码位于 `fmm.py` 中。

代码示例结果为：

```python
S = "他是研究生物化学的一位科学家"
vocab = ["他", "是", "研究", "研究生", "生物", "化学", "的", "一位", "科学", "科学家"]
print(partition(S, vocab))

# Output
['他', '是', '研究生', '物', '化学', '的', '一位', '科学家']
```

从这个结果中可以看出：
- 优点：
    - 程序简单易行，开发周期短；
    - 仅需要很少的语言资源（词表），不需要任何词法、句法、语义资源；
- 弱点：
    - 歧义消解的能力差；
    - 切分正确率不高，一般在95％左右；

#### 逆向最大匹配算法

逆向最大匹配算法的算法思想类似于正向最大匹配算法，算法的实现代码位于 `bmm.py` 中。

代码示例结果为：

```python
S = "他是研究生物化学的一位科学家"
vocab = ["他", "是", "研究", "研究生", "生物", "化学", "的", "一位", "科学", "科学家"]
print(partition(S, vocab))

# Output
['他', '是', '研究', '生物', '化学', '的', '一位', '科学家']
```

#### 双向最大匹配算法

双向最大匹配算法的原理就是将正向最大匹配算法和逆向最大匹配算法进行比较，从而选择正确的分词方式。

比较原则与步骤：

比较两种匹配算法的结果:

- 如果分词数量结果不同，则选择数量较少的那个
- 如果分词数量结果相同
    - 分词结果相同，返回任意一个
    - 分词结果不同，返回单字数较少的一个
    - 若单字数也相同，任意返回一个

双向最大匹配算法的代码位于 `bidirectionalmm.py` 中。

代码示例结果为：

```python
S = "他是研究生物化学的一位科学家"
vocab = ["他", "是", "研究", "研究生", "生物", "化学", "的", "一位", "科学", "科学家"]
print(partition(S, vocab))

# Output
['他', '是', '研究', '生物', '化学', '的', '一位', '科学家']
```

## 无词典切分

### 基于语言模型的分词方法

第四章

## 基于规则的方法

## 基于统计的方法

