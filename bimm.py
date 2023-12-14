from bmm import partition as bmm_partition
from fmm import partition as fmm_partition

from utils import get_single_chars_num

def partition(S, vocab):
    """
    Args:
        S (str): Input string with Chinese characters.
        vocab (list): The vocabulary that store the words.
    """
    fmm_results = fmm_partition(S, vocab)
    bmm_results = bmm_partition(S, vocab)

    if len(fmm_results) != len(bmm_results):
        return fmm_results if len(fmm_results) < len(bmm_results) else bmm_results
    else:
        if fmm_results == bmm_results:
            return fmm_results
        elif get_single_chars_num(fmm_results) != get_single_chars_num(bmm_results):
            return fmm_results if get_single_chars_num(fmm_results) < get_single_chars_num(bmm_results) else bmm_results
        else:
            return bmm_results
            

if __name__ == "__main__":
    S = "他是研究生物化学的一位科学家"
    vocab = ["他", "是", "研究", "研究生", "生物", "化学", "的", "一位", "科学", "科学家"]
    print(partition(S, vocab))
