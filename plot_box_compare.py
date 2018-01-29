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
args = parser.parse_args()

aggdat = pd.read_csv(args.input, sep='\t')
df1 = pd.melt(aggdat.ix[:, aggdat.columns != 'N_taxa'], id_vars=['method', 'lambda'], var_name='metric')
name_change = {'samm_rank': 'SAMM Rank', 'GCtree': 'GCtree', 'IgPhyML': 'IgPhyML', 'dnaml': 'dnaml', 'dnapars': 'dnapars', 'IQ-TREE-m000000': 'IQ-TREE JC', 'IQ-TREE-m010010': 'IQ-TREE HKY', 'IQ-TREE-m123450': 'IQ-TREE GTR'}
df1['method'] = df1['method'].replace(name_change)





palette_methods = {m:c for c, m in zip(sns.color_palette("Set2", 8), name_change.values())}

metrics = ['RF', 'MRCA', 'COAR']
outliers = [False, True]
lambda_ = ['x1/2', 'Normal', 'x2']


plot_iter = [(m, o) for o in outliers for m in metrics]
lambda_combi = [(lambda_[1], lambda_[0]), (lambda_[1], lambda_[2])]

o = False
with PdfPages(args.outbase+'_new.pdf') as pdf_pages:
    for m in metrics:
        for l in lambda_:
            plt.figure(figsize=(6, 3))
            m_df = df1[df1['metric'] == m]
            m_df = m_df[m_df['lambda'] == l]
            sort_m = m_df.groupby(['method']).mean().sort_values(by='value')
            order_m = list(sort_m.index)
            p = sns.boxplot(y="method", x="value", data=m_df, showfliers=o, showmeans=True, orient="h", order=order_m, palette=palette_methods)
            p.axes.set_title('Lambda = {}'.format(l), fontsize=15)
            p.set_ylabel('')
            p.set_xlabel('{} distance'.format(m))
            plt.axvline(sort_m.values[0], color='k', linestyle='--', linewidth=1.4)
            plt.axvline(sort_m.values[-1], color='k', linestyle='--', linewidth=1.4)
            plt.tight_layout()
            pdf_pages.savefig()

        m_df = df1[df1['metric'] == m]
        for lc in lambda_combi:
            plt.figure(figsize=(6, 6))
            x = m_df[m_df['lambda'] == lc[0]].groupby(['method']).mean()['value']
            y = m_df[m_df['lambda'] == lc[1]].groupby(['method']).mean()['value']
            color_list = [palette_methods[m_name] for m_name in list(m_df[m_df['lambda'] == lc[1]].groupby(['method']).mean().index)]
            p = sns.regplot(x, y, scatter_kws={'color':color_list})
            p.axes.set_title('Metric correlation ({})'.format(m), fontsize=15)
            p.set_xlabel('Lambda = {}'.format(lc[0]))
            p.set_ylabel('Lambda = {}'.format(lc[1]))
            plt.tight_layout()
            pdf_pages.savefig()

        plt.figure(figsize=(6, 6))
        m_df = df1[df1['metric'] == m]
        sort_m = m_df.groupby(['method']).mean().sort_values(by='value')
        order_m = list(sort_m.index)
        p = sns.boxplot(y="method", x="value", hue='lambda', data=m_df, showfliers=o, showmeans=True, orient="h", order=order_m, hue_order=lambda_)
        p.axes.set_title('Lambda comparison', fontsize=15)
        p.set_ylabel('')
        p.set_xlabel('{} distance'.format(m))
        plt.tight_layout()
        pdf_pages.savefig()




