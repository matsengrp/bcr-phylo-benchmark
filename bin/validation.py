#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''
comparison of inference and simulated trees
'''

from __future__ import division, print_function

import GCutils
from GCutils import CollapsedTree, CollapsedForest, hamming_distance
from COAR import COAR
try:
    import cPickle as pickle
except:
    import pickle
import pandas as pd
import scipy
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style='white', color_codes=True)
import os, sys
import numpy as np


def validate(true_tree, inferences, true_tree_colormap, outbase):
    '''
    inferences is a dict mapping inference name, like "gctree" to pickle files of
    CollapsedForest
    '''

    # With/without frequency weighting:
    all_COAR = lambda x, y: [COAR(x, y, freq_weigthing=fw) for fw in [False, True]]

    # if gctre is among the inferences, let's evaluate the likelihood ranking
    # among the parsimony trees
    ####### Turn off to avoid direct GCtree dependencies.
    """
    if 'gctree' in inferences:
        n_trees = len(inferences['gctree'].forest)
        # note: the unrooted_trees flag is needed because, for some reason, the RF
        #       function sometimes thinks the collapsed trees are unrooted and barfs
        distances, likelihoods = zip(*[(true_tree.compare(tree, method='RF'),
                                        tree.l(inferences['gctree'].params)[0]) for tree in inferences['gctree'].forest])
        MRCAs = [true_tree.compare(tree, method='MRCA') for tree in inferences['gctree'].forest]
        lineage_distances = [all_COAR(true_tree, tree) for tree in inferences['gctree'].forest]
        lineage_distances = zip(*lineage_distances)  # Unzip the forest tuple to get lineage_distances[ld0-3][tree_n]
        mean_frequencies = [scipy.mean([node.frequency for node in tree.tree.traverse()]) for tree in inferences['gctree'].forest]
        mean_branch_lengths = [scipy.mean([node.dist for node in tree.tree.iter_descendants()]) for tree in inferences['gctree'].forest]
        df = pd.DataFrame({'log-likelihood':likelihoods,
                           'RF':distances,
                           'MRCA':MRCAs,
                           'COAR':lineage_distances[0],
                           'COAR_fw':lineage_distances[1],
                           'mean_frequency':mean_frequencies,
                           'mean_branch_length':mean_branch_lengths})

        if n_trees > 1:
            # plots
            maxll = df['log-likelihood'].max()
            if len(df[df['log-likelihood']!=maxll]) >= 2:
                fit_reg = True
            else:
                fit_reg = False
            plt.figure(figsize=(10, 10))
            for i, metric in enumerate(('RF', 'MRCA', 'COAR', 'COAR_fw'), 1):
                plt.subplot(2, 2, i)
                ax = sns.regplot('log-likelihood', metric, data=df[df['log-likelihood']!=maxll], fit_reg=fit_reg, color='black', scatter_kws={'alpha':.8, 'clip_on':False})
                sns.regplot('log-likelihood', metric, data=df[df['log-likelihood']==maxll], fit_reg=False, color='red', scatter_kws={'alpha':.8, 'clip_on':False}, ax=ax)
                plt.ylim(0, 1.1*df[metric].max())
                plt.xlim(df['log-likelihood'].min(), df['log-likelihood'].max())
                plt.tight_layout()
            plt.savefig(outbase+'_gctree.pdf')

        df.to_csv(outbase+'.gctree.tsv', sep='\t', index=False)

        plt.figure(figsize=(14, 14))
        sns.pairplot(df, kind="reg")
        plt.savefig(outbase+'_pairplot.pdf')

        for i, tree in enumerate(inferences['gctree'].forest, 1):
            colormap = {}
            for node in tree.tree.traverse():
                if node.sequence in true_tree_colormap:
                    colormap[node.name] = true_tree_colormap[node.sequence]
                else:
                    assert node.frequency == 0
                    colormap[node.name] = 'lightgray'
            tree.render(outbase+'_gctree_colored_tree_{}.svg'.format(i), colormap=colormap)
    """



    # compare the inference methods
    # assume the first tree in the forest is the inferred tree

    methods, n_taxa, distances, MRCAs, lineage_distances = zip(
        *[(method,
           len(list(true_tree.tree.traverse())),  # Get all taxa in the tree
           true_tree.compare(inferences[method].forest[0], method='RF'),
           true_tree.compare(inferences[method].forest[0], method='MRCA'),
           all_COAR(true_tree, inferences[method].forest[0])) for method in inferences])
    lineage_distances = zip(*lineage_distances)  # Unzip the methods tuple to get lineage_distances[ld0-3][method]
    df = pd.DataFrame({'method':methods, 'N_taxa':n_taxa, 'RF':distances, 'MRCA':MRCAs, 'COAR':lineage_distances[0], 'COAR_fw':lineage_distances[1]},
                      columns=('method', 'N_taxa', 'RF', 'MRCA', 'COAR', 'COAR_fw'))
    df.to_csv(outbase+'.tsv', sep='\t', index=False)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='validate results of inference on simulation data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('true_tree', type=str, help='.p file containing true tree')
    parser.add_argument('true_tree_colormap', type=str, help='.tsv colormap file for true tree')
    parser.add_argument('forest_files', type=str, nargs='*', help='.p files containing forests from each inference method')
    parser.add_argument('--outbase', type=str, required=True, help='output file base name')
    args = parser.parse_args()

    with open(args.true_tree, 'rb') as f:
        true_tree = pickle.load(f)
    inferences = {}
    for forest_file in args.forest_files:
        with open(forest_file, 'rb') as f:
            forest = pickle.load(f)
            inferences[forest.name] = forest
    # now we rerender the inferred trees, but using colormap from true tree, makes visual comaprison easier
    true_tree_colormap = {} # map for tree sequences
    with open(args.true_tree_colormap, 'r') as f:
        for line in f:
            name, color = line.rstrip().split('\t')
            if ',' not in name:
                true_tree_colormap[true_tree.tree.search_nodes(name=name)[0].sequence] = color
            else:
                search = true_tree.tree.search_nodes(name=tuple(name.split(',')))
                if search:
                    true_tree_colormap[search[0].sequence] = color

    validate(true_tree, inferences, true_tree_colormap, args.outbase)
    print('Done')  # Print something to the log to make the scons wait_func run smoothly


if __name__ == '__main__':
    main()
