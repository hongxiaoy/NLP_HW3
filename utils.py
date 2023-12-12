def get_max_len_of_words(vocab):
    max_len = 0
    for w in vocab:
        if len(w) > max_len:
            max_len = len(w)
    return max_len


def get_single_chars_num(results):
    num = 0
    for f in results:
        if len(f) == 1:
            num += 1
    return num