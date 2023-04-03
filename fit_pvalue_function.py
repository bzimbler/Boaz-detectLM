"""
Script to read log-perplexity data of many sentences and characterize the empirical distribution.
We also report the mean log-perplexity as a function of sentence length
"""

import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
import logging
import numpy as np
from scipy.stats import norm, kurtosis
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib as mpl
mpl.style.use('ggplot')
import argparse

logging.basicConfig(level=logging.INFO)

from glob import glob

def read_all_csv_files(pattern):
    df = pd.DataFrame()
    for f in glob(pattern):
        df = pd.concat([df, pd.read_csv(f)])
    return df

def fit_pval_func(xx, G = 501):
    qq = np.linspace(0, 1, G)
    yy = [np.quantile(xx, q) for q in qq]
    return interp1d(yy, 1-qq, fill_value = (1,0), bounds_error=False)

def get_pval_func_dict(df, min_len=5, max_len=80):
    """
    One pvalue function for every length in the range(min_len, max_len)

    :param df:
    :param min_len:
    :param max_len:
    :return:
    """
    pval_func_list = [(c[0], fit_pval_func(c[1]['logloss']))
                      for c in df.groupby('length') if min_len <= c[0] <= max_len]
    return dict(pval_func_list)


def plot_histogram(datam):
    kurt = lambda x: kurtosis(x)
    agg_stats = datam.logloss.agg(['mean', 'median', 'std', 'skew'])#, kurt])
    print(agg_stats)
    tt = np.linspace(0, 10, 100)
    datam.logloss.plot.hist(bins=tt, density=True)
    plt.plot(tt, norm.pdf(tt, loc=agg_stats.loc['mean'],
                          scale=agg_stats.loc['std']),
             'r', alpha=0.5)


def plot_perp_vs_len(data, min_len=5, max_len=60):
    df_grouped = (
        data[['length', 'logloss']][(data['length'] >= min_len)
                                    & (data['length'] <= max_len)
                                    ].groupby(['length']).agg(['mean', 'std', 'count'])
    )
    df_grouped = df_grouped.droplevel(axis=1, level=0).reset_index()
    # Calculate a confidence interval as well.
    df_grouped['ci'] = 1.96 * df_grouped['std'] / np.sqrt(df_grouped['count'])
    df_grouped['ci_lower'] = df_grouped['mean'] - df_grouped['ci']
    df_grouped['ci_upper'] = df_grouped['mean'] + df_grouped['ci']
    df_grouped.head()

    fig, ax = plt.subplots()
    x = df_grouped['length']
    ax.plot(x, df_grouped['mean'])
    ax.fill_between(
        x, df_grouped['ci_lower'], df_grouped['ci_upper'], color='b', alpha=.15)
    ax.set_ylim(ymin=0)
    ax.set_title('log perplexity vs. length')
    ax.set_xlabel("Sentence Length [tokens]")
    ax.set_ylabel("Log Perplexity [nats]")
    ax.set_ylim((2, 4))
    plt.rcParams["figure.figsize"] = (8, 5)


def main():
    parser = argparse.ArgumentParser(description='Illustrate histogram and expected log-perplexity')
    parser.add_argument('-i', type=str, help='input file patten', default="results/gpt_sent_perp_*.csv")
    args = parser.parse_args()

    pattern = args.i
    logging.info(f"Reading null data from {pattern} and fitting survival function")
    df_null = read_all_csv_files(pattern)
    pval_functions = get_pval_func_dict(df_null)

    plt.figure()
    plot_histogram(df_null)
    plt.savefig('hist.png')
    plt.show()

    plt.figure()
    plot_perp_vs_len(df_null, min_len=5, max_len=40)
    plt.savefig("logperp_vs_len_gpt.png")
    plt.show()


if __name__ == '__main__':
    main()