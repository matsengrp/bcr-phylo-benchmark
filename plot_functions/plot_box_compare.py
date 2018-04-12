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
import pickle
import sys, os
import itertools
import seaborn as sns
from scipy.stats.stats import pearsonr
import scikits.bootstrap as sci
import numpy as np
from scipy.stats import pearsonr
from matplotlib.backends.backend_pdf import PdfPages
import math
sns.set_style("whitegrid")
#sns.set_style("ticks")
#sns.despine()
from matplotlib.ticker import FuncFormatter

def MyFormatter(x, lim):
    if x == 0:
        return 0
    return '{:.2E}'.format(x)
majorFormatter = FuncFormatter(MyFormatter)


parser = argparse.ArgumentParser(description='aggregate validation of repeated runs with same parameters')
parser.add_argument('input', type=str, help='validation.tsv files')
parser.add_argument('--outbase', type=str, help='output file base name')
parser.add_argument('--Nboot', type=int, default=10000, help='Number of bootstrap samples.')
global args
args = parser.parse_args()


metrics = ['RF', 'MRCA', 'COAR']
outliers = [False, True]

lambda_labs = ['x1/2', 'Normal', 'x2']
lambda_ = [0.1825, 0.365, 0.73]
name_change = {'samm_rank': 'SAMM Rank', 'GCtree': 'GCtree', 'IgPhyML': 'IgPhyML', 'dnaml': 'dnaml', 'dnapars': 'dnapars', 'IQ-TREE-m000000': 'IQ-TREE JC', 'IQ-TREE-m010010': 'IQ-TREE HKY', 'IQ-TREE-m123450': 'IQ-TREE GTR'}
palette_methods = {m:c for c, m in zip(sns.color_palette("Set2", 8), name_change.values())}

plot_iter = [(m, o) for o in outliers for m in metrics]
lambda_combi = [(lambda_[1], lambda_[0]), (lambda_[1], lambda_[2])]
lambda_combi_labs = [(lambda_labs[1], lambda_labs[0]), (lambda_labs[1], lambda_labs[2])]



if os.path.exists('df1_cache.p') and os.path.exists('ci_dict_cache.p'):
    df1 = pickle.load(open( "df1_cache.p", "rb" ))
    ci_dict = pickle.load(open( "ci_dict_cache.p", "rb" ))
else:
    aggdat = pd.read_csv(args.input, sep='\t')
    df1 = pd.melt(aggdat.ix[:, aggdat.columns != 'N_taxa'], id_vars=['method', 'lambda0'], var_name='metric')
    df1['method'] = df1['method'].replace(name_change)

    ci_dict = dict()
    for m in metrics:
        if m not in ci_dict:
            ci_dict[m] = dict()
        for l, l_ in zip(lambda_labs, lambda_):
            if l_ not in ci_dict[m]:
                ci_dict[m][l_] = dict()
            plt.figure(figsize=(6, 3))
            m_df = df1[df1['metric'] == m]
            m_df = m_df[m_df['lambda0'] == l_]
            m_df = m_df.drop(['lambda0', 'metric'], axis=1)

            for method in set(m_df['method']):
                df_slice = m_df[m_df['method'] == method]['value'].values
                Nsamples = len(df_slice)
                boots = np.array([np.mean(np.random.choice(df_slice, Nsamples, replace=True)) for i in range(args.Nboot)])
                boot_ci = (np.percentile(boots, 2.5), np.percentile(boots, 50), np.percentile(boots, 97.5))
                ci_dict[m][l_][method] = boot_ci

    pickle.dump(df1, open( "df1_cache.p", "wb" ))
    pickle.dump(ci_dict, open( "ci_dict_cache.p", "wb" ))




o = False
with PdfPages(args.outbase+'.pdf') as pdf_pages:
    for m in metrics:
        for l, l_ in zip(lambda_labs, lambda_):
            plt.figure(figsize=(6, 3))
            m_df = df1[df1['metric'] == m]
            m_df = m_df[m_df['lambda0'] == l_]
            sort_m = m_df.groupby(['method']).mean().sort_values(by='value')
            order_m = list(sort_m.index)
#            p = sns.boxplot(y="method", x="value", data=m_df, showfliers=o, showmeans=True, orient="h", order=order_m, palette=palette_methods, notch=False)
            p = sns.boxplot(y="method", x="value", data=m_df, showfliers=o, orient="h", order=order_m, palette=palette_methods, notch=False)
            p.axes.set_title('Lambda = {}'.format(l), fontsize=15)
            p.set_ylabel('')
            p.set_xlabel('{} distance'.format(m))
            plt.axvline(sort_m['value'].values[0], color='k', linestyle='--', linewidth=1.4)
            plt.axvline(sort_m['value'].values[-1], color='k', linestyle='--', linewidth=1.4)

            errors = [[abs(ci_dict[m][l_][method][1] - ci_dict[m][l_][method][i]) for method in order_m] for i in [0, 2]]
            x = [ci_dict[m][l_][method][1] for method in order_m]
            y = list(range(len(order_m)))
            plt.errorbar(x, y, xerr=errors, fmt = '^', color = '#B22222', barsabove=True, zorder=10)  # set zorder high to overwrite boxplot whiskers
            plt.tight_layout()
            pdf_pages.savefig()

        m_df = df1[df1['metric'] == m]
        for lcl, lc in zip(lambda_combi_labs, lambda_combi):
            plt.figure(figsize=(6, 6))
            x = m_df[m_df['lambda0'] == lc[0]].groupby(['method']).mean()['value']
            y = m_df[m_df['lambda0'] == lc[1]].groupby(['method']).mean()['value']
            method_list = list(m_df[m_df['lambda0'] == lc[1]].groupby(['method']).mean().index)
            color_list = [palette_methods[m_name] for m_name in method_list]
            p = sns.regplot(x, y, scatter_kws={'color':color_list, 'zorder':1}, ci=None)
            p.axes.set_title('Metric correlation ({})'.format(m), fontsize=15)
            #p.axes.xaxis.set_major_formatter(majorFormatter)
            #p.axes.yaxis.set_major_formatter(majorFormatter)
            Nticks = len(p.get_xticklabels())
            for ind, label in enumerate(p.get_xticklabels()):
                if (ind - (Nticks % 2)) % 2 == 0: # every 2th label is kept
                    label.set_visible(True)
                else:
                    label.set_visible(False)
            Nticks = len(p.get_yticklabels())
            for ind, label in enumerate(p.get_yticklabels()):
                if (ind - ((Nticks - 1) % 2)) % 2 == 0: # every 2th label is kept
                    label.set_visible(True)
                else:
                    label.set_visible(False)

            p.set_xlabel('Lambda = {}'.format(lcl[0]))
            p.set_ylabel('Lambda = {}'.format(lcl[1]))

            # Make scatter in the background to enable matplotlib legend:
            for t in zip(x, y, color_list, method_list):
                x_i, y_i, c, method = t
                plt.scatter(x=x_i, y=y_i, c=c, label=method, zorder=0)
            p.legend(loc=2)

            # Add text with pearson correlation:
            corr = math.floor(abs(pearsonr(x, y)[0]) * 100)/100.0
            text_pos = [(np.max(x) - np.min(x)) * 0.5 + np.min(x), (np.max(y) - np.min(y)) * 0.8 + np.min(y)]
            y_start, y_end = p.get_ylim()
            text_pos[1] = (y_end - y_start) * 0.9 + y_start
            plt.text(text_pos[0], text_pos[1], 'Pearson correlation: {}'.format(corr), bbox={'facecolor':'red', 'alpha':0.5, 'pad':5}, horizontalalignment='center', verticalalignment='center', fontsize=10);

            plt.tight_layout()
            pdf_pages.savefig()

        plt.figure(figsize=(6, 6))
        m_df = df1[df1['metric'] == m]
        sort_m = m_df.groupby(['method']).mean().sort_values(by='value')
        order_m = list(sort_m.index)
#        p = sns.boxplot(y="method", x="value", hue='lambda0', data=m_df, showfliers=o, showmeans=True, orient="h", order=order_m, hue_order=lambda_)
        p = sns.boxplot(y="method", x="value", hue='lambda0', data=m_df, showfliers=o, orient="h", order=order_m, hue_order=lambda_, palette=sns.color_palette()[::-1])
        handles, labels = p.get_legend_handles_labels()
        p.legend(handles, lambda_labs)
        p.axes.set_title('Lambda comparison', fontsize=15)
        p.set_ylabel('')
        p.set_xlabel('{} distance'.format(m))

        errors = [[abs(ci_dict[m][l_][method][1] - ci_dict[m][l_][method][i]) for method in order_m for l_ in lambda_] for i in [0, 2]]
        x = [ci_dict[m][l_][method][1] for method in order_m for l_ in lambda_]
        y = list()
        for i in range(len(order_m)):
            y.append(i - 0.267)
            y.append(i)
            y.append(i + 0.267)
        plt.errorbar(x, y, xerr=errors, fmt = '^', color = '#B22222', barsabove=True, zorder=10)  # set zorder high to overwrite boxplot whiskers
        plt.tight_layout()
        pdf_pages.savefig()

    comp_set = set([tuple(sorted((m1, m2))) for m1 in metrics for m2 in metrics if m1 != m2])
    for comp in sorted(comp_set):
        x_m, y_m = comp
        x = np.array([ci_dict[x_m][l_][method][1] for method in ci_dict[x_m][l_].keys() for l_ in ci_dict[x_m].keys()])
        y = np.array([ci_dict[y_m][l_][method][1] for method in ci_dict[y_m][l_].keys() for l_ in ci_dict[y_m].keys()])

        plt.figure(figsize=(6, 6))
        p = sns.regplot(x, y, ci=None)
        p.axes.set_title('Metric correlation ({} vs. {})'.format(x_m, y_m), fontsize=15)
        p.set_xlabel('Metric = {}'.format(x_m))
        p.set_ylabel('Metric = {}'.format(y_m))

        # Add text with pearson correlation:
        corr = math.floor(abs(pearsonr(x, y)[0]) * 100)/100.0
        text_pos = [(np.max(x) - np.min(x)) * 0.5 + np.min(x), (np.max(y) - np.min(y)) * 0.8 + np.min(y)]
        y_start, y_end = p.get_ylim()
        text_pos[1] = (y_end - y_start) * 0.9 + y_start
        plt.text(text_pos[0], text_pos[1], 'Pearson correlation: {}'.format(corr), bbox={'facecolor':'red', 'alpha':0.5, 'pad':5}, horizontalalignment='center', verticalalignment='center', fontsize=10);

        Nticks = len(p.get_xticklabels())
        for ind, label in enumerate(p.get_xticklabels()):
            if (ind - (Nticks % 2)) % 2 == 0: # every 2th label is kept
                label.set_visible(True)
            else:
                label.set_visible(False)
        Nticks = len(p.get_yticklabels())
        for ind, label in enumerate(p.get_yticklabels()):
            if (ind - ((Nticks - 1) % 2)) % 2 == 0: # every 2th label is kept
                label.set_visible(True)
            else:
                label.set_visible(False)

        plt.tight_layout()
        pdf_pages.savefig()


