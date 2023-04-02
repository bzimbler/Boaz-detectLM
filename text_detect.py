import pandas as pd
from multitest import MultiTest
from tqdm import tqdm
import logging


def to_sentences(text):
    return [str(s) for s in nlp(text).sents]


class ModelTextDetect(object):
    def __init__(self, model, tokenizer, survival_function_dict, logperp_function):
        """

        :param model:   Huggingface language model
        :param tokenizer:   Huggingface tokenizer
        :param survival_function_dict:  function to evaluate perplexity P-values
        :param logloss_function:  function to evaluate perplexity
        """
        self.model = model
        self.tokenizer = tokenizer
        self.survival_function_dict = survival_function_dict
        self.logperp_function = logperp_function
        self.min_len = 5
        self.max_len = 35

    def _logperp(self, sent: str, context=None) -> float:
        return float(self.logperp_function(self.model, self.tokenizer, sent, context))

    def _test_sent(self, sent: str, context=None) -> (float, float):
        """
        Returns:
          response:  sentence log-perplexity
          pvalue:
        """
        length = len(sent.split())
        if length >= self.min_len and length <= self.max_len:
            response = self._logperp(sent, context)
            pval = self.survival_function_dict[length](float(response))
            return response, pval
        else:
            return np.nan, np.nan

    def get_pvals(self, sentences: [str]):
        """
        Log-perplexity test of every sentecne
        """
        pvals = np.zeros(len(sentences))
        responses = np.zeros(len(sentences))
        for i, sent in tqdm(enumerate(sentences)):
            response, pval = self._test_sent(sent)
            pvals[i] = pval
            responses[i] = response
        return sentences, pvals, responses

    def testHC(self, sentences: [str]) -> float:
        pvals = self.get_pvals(sentences)[1]
        mt = MultiTest(pvals)
        return mt.hc()[0]

    def testFisher(self, sentences: [str]):
        pvals = self.get_pvals(sentences)[1]
        mt = MultiTest(pvals)
        return dict(zip(['Fn', 'pvalue'], mt.fisher()))

    def detect_stats(self, text_chunks: [str]):
        sentences, pvals, responses = self.get_pvals(text_chunks)

        df = pd.DataFrame({'sentence': sentences, 'x': responses, 'pval': pvals},
                          index=range(len(sentences)))
        mt = MultiTest(df[~df.pval.isna()].pval)
        hc, hct = mt.hc()
        df['mask'] = df['pval'] <= hct

        df['hc'] = hc
        df['fisher (pval)'] = mt.fisher()[1]

        return df
