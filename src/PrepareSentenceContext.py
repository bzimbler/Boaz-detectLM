import numpy as np
import pandas as pd
from multitest import MultiTest
from tqdm import tqdm
import torch
import logging
import spacy


class PrepareSentenceContext(object):
    """
    Parse text and extract length and context information

    This information is needed for evaluating log-perplexity of the text with respect to a language model
    and later on to test the likelihood that the sentence was sampled from the model with the relevant context.
    """

    def __init__(self, engine='spacy', context_policy=None,
                 context=None):
        if engine == 'spacy':
            self.nlp = spacy.load("en_core_web_sm")

        self.context_policy = context_policy
        self.context = context

    def __call__(self, text):
        return self.parse_text(text)

    def parse_text(self, text):
        texts = []
        contexts = []
        lengths = []
        previous = None

        parsed = self.nlp(text)
        for i, sent in enumerate(parsed.sents):
            lengths.append(len(sent))
            text = str(sent.sent)
            texts.append(text)

            if self.context is not None:
                context = self.context
            elif self.context_policy is None:
                context = None
            elif self.context_policy == 'previous_sentence':
                context = previous
                previous = text
            else:
                context = None

            contexts.append(context)
        return {'text': texts, 'length': lengths, 'context': contexts}
