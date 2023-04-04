import spacy
import numpy as np


def mix_lists(lst0: list, lst1: list, eps: float) -> dict:
    """
    Mix elements from lst0 and lst1 according to the mixture probability eps

    Params:
        :lst0   the base list
        :lst1   the mixture list
        :eps    the probability of replacing an element in lst0 by the corresponding element in lst1

    Returns:
        :mixted list
        :I   indexes of replaced elements
    """
    n = min(len(lst1), len(lst0))
    I = np.random.rand(n) < eps
    return [lst1[i] if i in I else lst0[i] for i in range(n)], I
