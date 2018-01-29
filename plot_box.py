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
df1 = pd.melt(aggdat.ix[:, aggdat.columns != 'N_taxa'], id_vars=['method'], var_name='metric')
name_change = {'samm_rank': 'SAMM Rank', 'GCtree': 'GCtree', 'IgPhyML': 'IgPhyML', 'dnaml': 'dnaml', 'dnapars': 'dnapars', 'IQ-TREE-m000000': 'IQ-TREE JC', 'IQ-TREE-m010010': 'IQ-TREE HKY', 'IQ-TREE-m123450': 'IQ-TREE GTR'}
df1['method'] = df1['method'].replace(name_change)

palette_methods = {m:c for c, m in zip(sns.color_palette("Set2", 8), name_change.values())}

metrics = ['RF', 'MRCA', 'COAR']
outliers = [False, True]

plot_iter = [(m, o) for o in outliers for m in metrics]
with PdfPages(args.outbase+'_new.pdf') as pdf_pages:
    for t in plot_iter:
        plt.figure(figsize=(6, 3))
        m, o = t
        m_df = df1[df1['metric'] == m]
        sort_m = m_df.groupby(['method']).mean().sort_values(by='value')
        order_m = list(sort_m.index)
        p = sns.boxplot(y="method", x="value", data=m_df, showfliers=o, showmeans=True, orient="h", order=order_m, palette=palette_methods)
        p.axes.set_title('Metric = {}'.format(m), fontsize=15)
        p.set_ylabel('')
        p.set_xlabel('{} distance'.format(m))
        plt.tight_layout()
        pdf_pages.savefig()


# Old vertical boxplots:
'''
print('Starting to plot')
plt.figure()

with PdfPages(args.outbase+'_box.pdf') as pdf_pages:
    p = sns.factorplot(x="method", y="value", col="metric", col_wrap=2,
                   data=df1, kind="box", size=6, aspect=.65, sharey=False, showfliers=False, showmeans=True)
    p.set_xticklabels(rotation=30)
    pdf_pages.savefig(p.fig)

    p = sns.factorplot(x="method", y="value", col="metric", col_wrap=2,
                   data=df1, kind="box", size=6, aspect=.65, sharey=False, showfliers=True, showmeans=True)
    p.set_xticklabels(rotation=30)
    pdf_pages.savefig(p.fig)
'''



