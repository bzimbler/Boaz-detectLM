"""
Apply the atomic chunk detector many times.
This is useful for:
 1. Characterizing the null distribution of a model with a specific context policy.
 2. Characterizing the power of the global detector against a mixtures from a specific domain.

"""

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import argparse
from src.PerplexityEvaluator import PerplexityEvaluator
from src.PrepareSentenceContext import PrepareSentenceContext
from datasets import load_dataset
from glob import glob

logging.basicConfig(level=logging.INFO)


def process_text(text, atomic_detector, parser):
    chunks = parser(text)

    ids = []
    lengths = []
    responses = []
    context_lengths = []
    chunk_num = 0
    for chunk, context, length in zip(chunks['text'], chunks['context'], chunks['length']):
        chunk_num += 1
        res = atomic_detector(chunk, context)
        ids.append(chunk_num)
        lengths.append(length)
        responses.append(res)
        if context:
            context_lengths.append(len(context.split()))
        else:
            context_lengths.append(0)

    return dict(chunk_ids=ids, responses=responses, lengths=lengths, context_lengths=context_lengths)


def iterate_over_texts(texts, atomic_detector, parser, output_file):
    ids = []
    lengths = []
    responses = []
    context_lengths = []
    names = []
    for name, text in tqdm(texts):
        try:
            r = process_text(text, atomic_detector, parser)
        except KeyboardInterrupt:
            break
        except:
            print("Error processing text")
            continue
        ids += r['chunk_ids']
        responses += r['responses']
        lengths += r['lengths']
        context_lengths += r['context_lengths']
        names += [name] * len(r['chunk_ids'])

        df = pd.DataFrame({'num': ids, 'length': lengths,
                           'response': responses, 'context_length': context_lengths,
                           'name': names})
        logging.info(f"Saving results to {output_file}")
        df.to_csv(output_file)


def get_text_data_from_files(path, extension='*.txt'):
    logging.info(f"Reading text data from {path}...")
    lo_fns = glob(path + extension)
    for fn in lo_fns:
        logging.info(f"Reading text from {fn}")
        with open(fn, "rt") as f:
            yield fn, f.read()


def get_text_from_wiki_dataset(author='machine'):
    dataset = load_dataset("aadityaubhat/GPT-wiki-intro")
    for d in tqdm(dataset['train']):
        machine_text = d['prompt'] + d['generated_text']
        text = machine_text if author == 'machine' else d['wiki_intro']
        name = d['title']
        yield name, text


def get_text_from_chatgpt_news_dataset(author='machine', n=0):
    dataset = load_dataset("isarth/chatgpt-news-articles")
    for d in tqdm(list(dataset['train'])[n:]):
        text = d['chatgpt'] if author == 'machine' else d['article']
        name = d['id']
        yield name, text


def get_text_from_wikibio_dataset(author='machine'):
    dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination")
    # features: ['gpt3_text', 'wiki_bio_text', 'gpt3_sentences', 'annotation', 'wiki_bio_test_idx'],
    for d in tqdm(dataset['train']):
        text = d['gpt3_text'] if author == 'machine' else d['wiki_bio_text']
        name = d['wiki_bio_test_idx']
        yield name, text


def main():
    parser = argparse.ArgumentParser(description='Apply atomic detector many times to characterize distribution')
    parser.add_argument('-i', type=str, help='data file', default="")
    parser.add_argument('--context', action='store_true')
    parser.add_argument('--human', action='store_true')
    args = parser.parse_args()

    lm_name = "gpt2"
    if args.context:
        context_policy = 'previous_sentence'
    else:
        context_policy = 'no_context'

    logging.debug(f"Loading Language model {lm_name}...")
    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    model = AutoModelForCausalLM.from_pretrained(lm_name)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.to(device)

    dataset_name = args.i

    author = 'human' if args.human else 'machine'

    if args.i == "wiki":
        logging.info("Processing wiki dataset...")
        texts = get_text_from_wiki_dataset(author)
    elif args.i == "wikibio":
        logging.info("Processing wiki bio dataset...")
        texts = get_text_from_wikibio_dataset(author)
    elif args.i == 'news':
        logging.info("Processing wiki bio dataset...")
        texts = get_text_from_chatgpt_news_dataset(author)
    else:
        texts = get_text_data_from_files(args.i, extension='*.txt')
        dataset_name = 'files'

    out_filename = f"{lm_name}_{context_policy}_{dataset_name}_{author}.csv"
    logging.info(f"Iterating over texts...")
    sentence_detector = PerplexityEvaluator(model, tokenizer)
    parser = PrepareSentenceContext(context_policy=context_policy)


    print(f"Saving results to {out_filename}")
    iterate_over_texts(texts, sentence_detector, parser, output_file=out_filename)


if __name__ == '__main__':
    main()
