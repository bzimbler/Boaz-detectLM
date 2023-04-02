import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, GPT2LMHeadModel
import logging

logging.basicConfig(level=logging.INFO)


def logloss(model, tokenizer, text, context=None, ignore_index=-100):
    text_ids = tokenizer(text, return_tensors='pt')
    if context:
        context_ids = tokenizer(context, return_tensors='pt')
        input_ids = torch.concatenate([context_ids['input_ids'], text_ids['input_ids']], axis=1)
        labels = torch.concatenate([torch.ones_like(context_ids['input_ids']) * (ignore_index), text_ids['input_ids']],
                                   axis=1)
    else:
        input_ids = text_ids['input_ids']
        labels = input_ids

    loss = model(input_ids=input_ids.to(device), labels=labels.to(device)).loss
    return loss.cpu().detach().numpy()


def eval_logloss_per_sentence(sentences, contexts=None, min_len=5, max_len=40):
    texts = []
    lengths = []
    loglosses = []
    sent_nums = []

    context = None
    for text in tqdm(sentences):
        parsed = nlp(text)
        sent_num = 0
        for i, sent in enumerate(parsed.sents):
            sent_num += 1
            if min_len <= len(sent) <= max_len:
                r = logloss(gpt_model, gpt_tokenizer, str(sent))

                contexts.append(context)
                texts.append(sent)
                lengths.append(len(sent))
                loglosses.append(float(r))
                sent_nums.append(sent_num)

    return pd.DataFrame(dict(text=texts, length=lengths, logloss=loglosses,
                             sent_num=sent_nums, context=contexts))


import spacy
from datasets import load_dataset

logging.debug("Loading Spacy...")
nlp = spacy.load("en_core_web_sm")

logging.debug("Loading Language model...")
gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt_model = AutoModelForCausalLM.from_pretrained("gpt2")

logging.debug("Loading Language model...")
dataset = load_dataset("aadityaubhat/GPT-wiki-intro")

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
gpt_model.to(device)

logging.info("Evaluating perplexities of texts...")
texts_gen = []
texts_wiki = []
batch_size = 150000
for i in range(batch_size):
    r = dataset['train'][i]
    texts_gen.append(r['generated_intro'])
    texts_wiki.append(r['wiki_intro'])

output_dir = "results/"
no_batch = 10
batch_size = 15000
for i in tqdm(range(4, batch_size * no_batch)):
    df_gpt = eval_logloss_per_sentence(texts_gen[i * batch_size:(i + 1) * batch_size])
    df_gpt.to_csv(output_dir + f"/gpt_sent_perp_{i}.csv")
