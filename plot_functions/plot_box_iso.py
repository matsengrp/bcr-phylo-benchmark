#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''
aggregation plots of validation output from several simulation/validation runs
'''

from __future__ import division, print_function
import scipy, matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from matplotlib import rc, ticker
import pandas as pd
import argparse
import sys
import numpy as np
import itertools
import seaborn as sns
from scipy.stats.stats import pearsonr
from matplotlib.backends.backend_pdf import PdfPages
sns.set_style("whitegrid")
#sns.set_style("ticks")
#sns.despine()

parser = argparse.ArgumentParser(description='aggregate validation of repeated runs with same parameters')
parser.add_argument('input', type=str, help='validation.tsv files')
parser.add_argument('--outbase', type=str, help='output file base name')
parser.add_argument('--Nboot', type=int, default=10000, help='Number of bootstrap samples.')
global args
args = parser.parse_args()

aggdat = pd.read_csv(args.input, sep='\t')
df1 = pd.melt(aggdat.ix[:, aggdat.columns != 'N_taxa'], id_vars=['method'], var_name='metric')
name_change = {'samm_rank': 'SAMM Rank', 'GCtree': 'GCtree', 'IgPhyML': 'IgPhyML', 'dnaml': 'dnaml', 'dnapars': 'dnapars', 'IQ-TREE-m000000': 'IQ-TREE JC', 'IQ-TREE-m010010': 'IQ-TREE HKY', 'IQ-TREE-m123450': 'IQ-TREE GTR'}
df1['method'] = df1['method'].replace(name_change)
metric_rename = {'iso_error1':'Isotype misplacements', 'iso_error2':'Isotype misplacements normalized'}
df1['metric'] = df1['metric'].replace(metric_rename)

palette_methods = {m:c for c, m in zip(sns.color_palette("Set2", 8), name_change.values())}

metrics = ['Isotype misplacements', 'Isotype misplacements normalized']
outliers = [False, True]

plot_iter = [(m, o) for o in outliers for m in metrics]
with PdfPages(args.outbase+'.pdf') as pdf_pages:
    for t in plot_iter:
        plt.figure(figsize=(6, 3))
        m, o = t
        m_df = df1[df1['metric'] == m]
        sort_m = m_df.groupby(['method']).mean().sort_values(by='value')
        order_m = list(sort_m.index)
#        p = sns.boxplot(y="method", x="value", data=m_df, showfliers=o, showmeans=True, orient="h", order=order_m, palette=palette_methods)
        p = sns.boxplot(y="method", x="value", data=m_df, showfliers=o, orient="h", order=order_m, palette=palette_methods)
        p.axes.set_title('Metric = {}'.format(m), fontsize=12)
        p.set_ylabel('')
        p.set_xlabel('{} distance'.format(m))
        plt.axvline(sort_m.values[0], color='k', linestyle='--', linewidth=1.4)
        plt.axvline(sort_m.values[-1], color='k', linestyle='--', linewidth=1.4)

        errors = [[], []]
        x = list()
        for method in order_m:
            df_slice = m_df[m_df['method'] == method]['value'].values
            Nsamples = len(df_slice)
            boots = np.array([np.mean(np.random.choice(df_slice, Nsamples, replace=True)) for i in range(args.Nboot)])
            errors[0].append(np.percentile(boots, 2.5))
            errors[1].append(np.percentile(boots, 97.5))
            x.append(np.percentile(boots, 50))

        y = list(range(len(order_m)))
        plt.errorbar(x, y, xerr=errors, fmt = '^', color = '#B22222', barsabove=True, zorder=10)  # set zorder high to overwrite boxplot whiskers
        plt.tight_layout()
        pdf_pages.savefig()


        # Make boxes of the differences to the best:
        plt.figure(figsize=(6, 3))

        diffs = list()
        errors = [[], []]
        x = list()
        for method in order_m[1:]:
            df_slice = m_df[m_df['method'] == method]['value'].values - m_df[m_df['method'] == order_m[0]]['value'].values
            diffs.append(df_slice.copy())
            Nsamples = len(df_slice)
            boots = np.array([np.mean(np.random.choice(df_slice, Nsamples, replace=True)) for i in range(args.Nboot)])
            mid = np.percentile(boots, 50)
            errors[0].append(abs(mid - np.percentile(boots, 2.5)))
            errors[1].append(abs(mid - np.percentile(boots, 97.5)))
            x.append(mid)

        diff_df = pd.DataFrame(list(zip(*diffs)), columns=order_m[1:])
        diff_df = pd.melt(diff_df, value_vars=order_m[1:], var_name='method')

        p = sns.boxplot(y="method", x="value", data=diff_df, showfliers=o, orient="h", order=order_m[1:], palette=palette_methods)
        p.axes.set_title('Metric = {}, diff to {}'.format(m, order_m[0]), fontsize=12)
        p.set_ylabel('')
        p.set_xlabel('{} difference'.format(m))

        y = list(range(len(order_m[1:])))
        plt.errorbar(x, y, xerr=errors, fmt = '^', color = '#B22222', barsabove=True, zorder=10)  # set zorder high to overwrite boxplot whiskers

        plt.tight_layout()
        pdf_pages.savefig()



