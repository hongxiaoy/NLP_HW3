from utils import get_max_len_of_words


def partition(S, vocab):
    """
    Args:
        S (str): Input string with Chinese characters.
        vocab (list): The vocabulary that store the words.
    """
    results = []

    i = 0
    while True:
        n = len(S) - i
        if n == 0:
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
        

if __name__ == "__main__":
    S = "他是研究生物化学的一位科学家"
    vocab = ["他", "是", "研究", "研究生", "生物", "化学", "的", "一位", "科学", "科学家"]
    print(partition(S, vocab))
