import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.interpolate import interp1d
import logging
import numpy as np
import argparse
from src.DetectLM import DetectLM
from src.PerplexityEvaluator import PerplexityEvaluator
from src.PrepareSentenceContext import PrepareSentenceContext
from glob import glob
import pathlib
import yaml
import re

logging.basicConfig(level=logging.INFO)


def read_all_csv_files(pattern):
    df = pd.DataFrame()
    print(pattern)
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

    :param df:  data frame with columns 'response' and 'length'
    :param min_len:  optional cutoff value
    :param max_len:  optional cutoff value
    :return:
    """
    assert not df.empty
    value_name = "response" if "response" in df.columns else "logloss"
    pval_func_list = [(c[0], fit_pval_func(c[1][value_name]))
                      for c in df.groupby('length') if min_len <= c[0] <= max_len]
    return dict(pval_func_list)


def mark_edits_remove_tags(chunks, tag="edit"):
    text_chunks = chunks['text']
    edits = []
    for i,text in enumerate(text_chunks):
        chunk_text = re.findall(rf"<{tag}>(.+)</{tag}>", text)
        if len(chunk_text) > 0:
            import pdb; pdb.set_trace()
            chunks['text'][i] = chunk_text[0]
            chunks['length'][i] -= 2
            edits.append(True)
        else:
            edits.append(False)

    return chunks, edits

def main():
    parser = argparse.ArgumentParser(description='Test document for non-model sentences')
    parser.add_argument('-i', type=str, help='input text file', default="input file")
    parser.add_argument('-conf', type=str, help='configurations file', default="conf.yml")
    parser.add_argument('--context', action='store_true')
    args = parser.parse_args()

    with open(args.conf, "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    print("context = ", args.context)

    if args.context:
        null_data_file = params['context-null-data-file']
    else:
        null_data_file = params['no-context-null-data-file']
    lm_name = params['language-model-name']

    logging.info(f"Using null data from {null_data_file} and fitting survival function")
    logging.info(f"Please verify that null data was obtained with the same context")
    logging.info(f"policy used in inference.")

    df_null = read_all_csv_files(null_data_file)

    if params['ignore-first-sentence']:
        df_null = df_null[df_null.num > 1]
    pval_functions = get_pval_func_dict(df_null)

    max_len = np.max(list(pval_functions.keys()))

    logging.info(f"Loading model and detection function...")

    logging.debug(f"Loading Language model {lm_name}...")
    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    model = AutoModelForCausalLM.from_pretrained(lm_name)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.to(device)

    if args.context:
        context_policy = 'previous_sentence'
    else:
        context_policy = None

    sentence_detector = PerplexityEvaluator(model, tokenizer)
    logging.debug("Initializing detector...")
    detector = DetectLM(sentence_detector, pval_functions,
                        max_len=max_len, length_limit_policy='truncate',
                        ignore_first_sentence=
                        True if context_policy == 'previous_sentence' else False
                        )

    input_file = args.i
    logging.info(f"Parsing document {input_file}...")

    if pathlib.Path(input_file).suffix == '.csv':
        text = pd.read_csv(input_file)
        engine = 'pandas'
    elif pathlib.Path(input_file).suffix == '.txt':
        engine = 'spacy'
        with open(input_file, 'rt') as f:
            text = f.read()
    else:
        logging.error("Unknown file extension")
        exit(1)

    parser = PrepareSentenceContext(engine=engine, context_policy=context_policy)
    chunks = parser(text)

    logging.info("Testing parsed document")
    res = detector(chunks['text'], chunks['context'])

    df = res['sentences']

    df['tag'] = chunks['tags']
    df.loc[df.tag.isna(), 'tag'] = 'not edit'

    print(f"Edit rate = {np.mean(df['tag'] == '<edit>')}")

    output_file = "out.csv"
    df.to_csv(output_file)

    print(df.groupby('tag').response.mean())
    print(df)
    print(f"HC = {res['HC']}")
    print(f"Fisher = {res['fisher']}")
    print(f"Fisher (chisquared pvalue) = {res['fisher_pvalue']}")


if __name__ == '__main__':
    main()
