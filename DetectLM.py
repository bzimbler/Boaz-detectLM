import numpy as np
import pandas as pd
from multitest import MultiTest
from tqdm import tqdm

def truncae_to_max_no_tokens(text, max_no_tokens):
    return " ".join(text.split()[:max_no_tokens])

class DetectLM(object):
    def __init__(self, sentence_detection_function, survival_function_dict,
                 context_policy: str = None, min_len=5, max_len=60,
                 length_limit_policy='truncate'):
        """
        Test for the presence of sentences of irregular origin as reflected by the
        sentence_detection_function. This function can be assisted by a context, which we
        determine using the context_policy argument.

        :param sentence_detection_function:  a function returning the log-perplexity of the text
        based on a candidate language model
        :param survival_function_dict:  survival_function_dict(x, l) is the probability of the language
        model to produce a sentence of log-perplexity as extreme as x or more, for an input sentence s
        of length l or a for an input pair (s, c) with sentence s of length l under context c.
        :param context_policy: how to determine the context. CURRENTLY NOT IMPLEMENTED.
        :param length_limit_policy: what should we do if a sentence is too long. Options are:
            'truncate':  truncate sentence to the maximal length :max_len
             'ignore':  do not evalaute the response and P-value for this sentence
             'max_available':  use the log-perplexity function of the maximal available length
        """

        self.survival_function_dict = survival_function_dict
        self.sentence_detector = sentence_detection_function
        self.min_len = min_len
        self.max_len = max_len
        self.context_policy = context_policy
        self.length_limit_policy = length_limit_policy

    def _logperp(self, sent: str, context=None) -> float:
        return float(self.sentence_detector(sent, context))

    def _test_sent(self, sent: str, context=None) -> (float, float):
        """
        Returns:
          response:  sentence log-perplexity
          pval:      P-value of atomic log-perplexity test
        """
        length = len(sent.split())  # This is the approximate. The precise length is determined by the tokenizer
        if self.min_len <= length:
            if length > self.max_len:  # in case length exceeds specifications...
                if self.length_limit_policy == 'truncate':
                    sent = truncae_to_max_no_tokens(sent, self.max_len)
                    length = self.max_len
                elif self.length_limit_policy == 'ignore':
                    return np.nan, np.nan
                elif self.length_limit_policy == 'max_available':
                    length = self.max_len
            response = self._logperp(sent, context)
            pval = self.survival_function_dict[length](float(response))
            return response, pval
        else:
            return np.nan, np.nan

    def get_pvals(self, sentences: [str], contexts: [str]) -> ([str], [float], [float]):
        """
        Log-perplexity test of every (sentence, context) pair
        """
        assert len(sentences) == len(contexts)

        pvals = np.zeros(len(sentences))
        responses = np.zeros(len(sentences))
        for i, (sent, ctx) in tqdm(enumerate(zip(sentences, contexts))):
            response, pval = self._test_sent(sent, ctx)
            pvals[i] = pval
            responses[i] = response
        return pvals, responses

    def testHC(self, sentences: [str]) -> float:
        pvals = self.get_pvals(sentences)[1]
        mt = MultiTest(pvals)
        mt.hc()[0]

    def testFisher(self, sentences: [str]) -> dict:
        pvals = self.get_pvals(sentences)[1]
        mt = MultiTest(pvals)
        return dict(zip(['Fn', 'pvalue'], mt.fisher()))


    def _test_chunked_doc(self, lo_chunks: [str], lo_contexts: [str]) -> MultiTest:
        pvals, responses = self.get_pvals(lo_chunks, lo_contexts)
        df = pd.DataFrame({'sentence': lo_chunks, 'response': responses, 'pvalue': pvals,
                           'context': lo_contexts},
                          index=range(len(lo_chunks)))
        return MultiTest(df[~df.pvalue.isna()].pvalue)

    def test_chunked_doc(self, lo_chunks: [str], lo_contexts: [str]) -> pd.DataFrame:
        mt = self._test_chunked_doc(lo_chunks, lo_contexts)
        hc, hct = mt.hc()
        fisher = mt.fisher()
        df['mask'] = df['pvalue'] <= hct
        return dict(sentences=df, HC=hc, fisher=fisher[0], fisher_pvalue=fisher[1])

    def test_chunked_doc_hc_dashboard(self, lo_chunks: [str], lo_contexts: [str]) -> pd.DataFrame:
        mt = self._test_chunked_doc(lo_chunks, lo_contexts)
        hc_rep = mt.hc_dashboard()
        return hc_rep

    def __call__(self, lo_chunks: [str], lo_contexts: [str]) -> pd.DataFrame:
        return self.test_chunked_doc(lo_chunks, lo_contexts)

