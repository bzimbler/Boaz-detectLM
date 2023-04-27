import logging
import spacy
import re


class Sentences(object):
    def __init__(self, texts):
        def iterate(texts):
            for t in texts:
                yield t

        self.sents = iterate(texts)


class PandasParser(object):
    """
    Iterate over the text column of a dataframe
    """

    def __init__(self, text_value='text'):
        self.text_value = text_value
        self.sents = None

    def __call__(self, df):
        texts = list(df[self.text_value])
        return Sentences(texts)



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
        if engine == 'pandas':
            self.nlp = PandasParser()

        self.context_policy = context_policy
        self.context = context

    def __call__(self, text):
        return self.parse_text(text)

    def parse_text(self, text):
        texts = []
        contexts = []
        lengths = []
        tags = []
        previous = None

        text = re.sub("(</?[a-zA-Z0-9 ]+>)\s+", r"\1. ", text)  # to make sure that tags are in separate sentences
        parsed = self.nlp(text)

        tag = None
        for i, sent in enumerate(parsed.sents):
            tag_text = re.findall(r"(</?[a-zA-Z0-9 ]+>)", str(sent))
            if len(tag_text) > 0:
                if tag is None: # opening tag
                    tag = tag_text[0]
                else:  # closing tag
                    tag = None

            else:  # only continue if text is not a tag
                tags.append(tag)
                lengths.append(len(sent))
                sent_text = str(sent)
                texts.append(sent_text)

                if self.context is not None:
                    context = self.context
                elif self.context_policy is None:
                    context = None
                elif self.context_policy == 'previous_sentence':
                    context = previous
                    previous = sent_text
                else:
                    context = None

                contexts.append(context)
        return {'text': texts, 'length': lengths, 'context': contexts, 'tag': tags}

