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
import seaborn as sns
from scipy.stats.stats import pearsonr
from matplotlib.backends.backend_pdf import PdfPages

parser = argparse.ArgumentParser(description='aggregate validation of repeated runs with same parameters')
parser.add_argument('input', type=str, help='validation.tsv files')
parser.add_argument('--outbase', type=str, help='output file base name')
args = parser.parse_args()

aggdat = pd.read_csv(args.input, sep='\t')
df1 = pd.melt(aggdat.ix[:, aggdat.columns != 'N_taxa'], id_vars=['method'], var_name='metric')

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




