import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.interpolate import interp1d
import logging
import numpy as np
import argparse
from DetectLM import DetectLM
from PerplexityEvaluator import PerplexityEvaluator
from PrepareSentenceContext import PrepareSentenceContext
from glob import glob

logging.basicConfig(level=logging.INFO)


def read_all_csv_files(pattern):
    df = pd.DataFrame()
    for f in glob(pattern):
        df = pd.concat([df, pd.read_csv(f)])
    return df

def fit_pval_func(xx, G=501):
    qq = np.linspace(0, 1, G)
    yy = [np.quantile(xx, q) for q in qq]
    return interp1d(yy, 1 - qq, fill_value=(1, 0), bounds_error=False)

def get_pval_func_dict(df, min_len=0, max_len=100):
    """
    One pvalue function for every length in the range(min_len, max_len)

    :param df:  data frame with columns 'logloss' and 'length'
    :param min_len:  optional cutoff value
    :param max_len:  optional cutoff value
    :return:
    """
    pval_func_list = [(c[0], fit_pval_func(c[1]['logloss']))
                      for c in df.groupby('length') if min_len <= c[0] <= max_len]
    return dict(pval_func_list)


def main():
    parser = argparse.ArgumentParser(description='Illustrate histogram and expected log-perplexity')
    parser.add_argument('-null', type=str, help='input pattern', default="results/gpt_sent_perp_*.csv")
    parser.add_argument('-i', type=str, help='input text file', default="input file")
    args = parser.parse_args()

    pattern = args.null
    logging.info(f"Reading null data from {pattern} and fitting survival function")
    df_null = read_all_csv_files(pattern)
    pval_functions = get_pval_func_dict(df_null)

    max_len = np.max(list(pval_functions.keys()))

    logging.info(f"Loading model and detection function...")

    lm_name = "gpt2"
    logging.debug(f"Loading Language model {lm_name}...")
    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    model = AutoModelForCausalLM.from_pretrained(lm_name)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.to(device)

    sentence_detector = PerplexityEvaluator(model, tokenizer)
    logging.debug("Initializing detector...")
    detector = DetectLM(sentence_detector, pval_functions, context_policy=None,
                        max_len=max_len, length_limit_policy='truncate')

    input_file = args.i
    logging.info(f"Parsing text from {input_file}...")
    with open(input_file, 'rt') as f:
        text = f.read()

    parse_chunks = PrepareSentenceContext(context_policy='previous_sentence')
    chunks = parse_chunks(text)

    logging.info("Testing parsed document")
    res = detector(chunks['text'], chunks['context'])  # ignore title

    print(res['sentences'])
    print(f"HC = {res['HC']}")
    print(f"Fisher (pvalue) = {res['fisher_pvalue']}")


if __name__ == '__main__':
    main()
