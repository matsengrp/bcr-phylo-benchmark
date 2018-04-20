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
from random import shuffle

global ISO_TYPE_ORDER
#ISO_TYPE_ORDER = [set(['IgM', 'IgD']), set(['IgG', 'IgGA', 'IgGb']), set(['IgE']), set(['IgA'])]
ISO_TYPE_ORDER = {'IgM': 1, 'IgD': 1, 'IgG': 2, 'IgGA': 2, 'IgGb': 2, 'IgE': 3, 'IgA': 4}



def count_misplacements(tree):
    misplaced = 0
    # Iterate leaves:
    for leaf in tree.iter_leaves():
        iso_order_child = min([ISO_TYPE_ORDER[iso] for iso in leaf.isotype])
        while leaf.up:
            # Isotypes are inherited by decendents if internal node is infered:
            if hasattr(leaf, 'isotype'):
                iso_order_child = min([ISO_TYPE_ORDER[iso] for iso in leaf.isotype])
            if hasattr(leaf.up, 'isotype'):
                iso_order_parent = min([ISO_TYPE_ORDER[iso] for iso in leaf.up.isotype])
            else:
                iso_order_parent = iso_order_child
            # Going down the tree isotype order must either increase or stay constant,
            # otherwise there is a misplacement:
            if iso_order_child < iso_order_parent:
                leaf.up.add_features(misplacement=True)
            else:
                leaf.up.add_features(misplacement=False)
            leaf = leaf.up
    return sum([1 for n in tree.traverse() if hasattr(n, 'misplacement') and n.misplacement is True])



def validate(heavy, light, outbase):
    '''
    Validate the inferred trees' consistency with experimental results
    such as isotype and chain pairing.
    '''
    methods, n_taxa = zip(*[(hf.name, len([n.name for n in hf.forest[0].tree.traverse() if n.frequency > 0])) for hf in heavy])
    iso_error1 = list()
    iso_error2 = list()
    for f in heavy:
        ie1 = 9999999999999
        ie2 = 9999999999999
        for ftree in f.forest:
            tree = ftree.tree


            #### Introduce an IgA just above the root (for debugging):
            #node_over_naive = list(tree.traverse())[2]
            #node_over_naive.add_feature('isotype', set(['IgA']))
            #print(node_over_naive)
            ####
            misplaced = count_misplacements(tree)
            Ntips = len(list(tree.iter_leaves()))
            misplaced_baseline = 0
            baseline_iterations = 10000
            # Do 10000 iterations of label shuffling to calculate
            # the expected number of misplacements with random label order:
            for i in range(baseline_iterations):
                tree_cp = tree.copy(method='deepcopy')
                iso_list = [n.isotype for n in tree_cp.iter_descendants() if hasattr(n, 'isotype')]
                shuffle(iso_list)
                for n in tree_cp.iter_descendants():
                    if hasattr(n, 'isotype'):
                        n.isotype = iso_list.pop()
                assert(len(iso_list) == 0)
                sc = count_misplacements(tree_cp)
                misplaced_baseline += sc
            misplaced_baseline /= baseline_iterations
            tree.add_feature('isotype_misplacement_score', misplaced / misplaced_baseline)

            if f.name != 'dnapars':
                ie1 = misplaced
                ie2 = misplaced / misplaced_baseline
                break
            elif (misplaced / misplaced_baseline) < ie2:
                #print('Before: {} and {}'.format(ie1, ie2))
                ie1 = misplaced
                ie2 = misplaced / misplaced_baseline
                #print('After: {} and {}'.format(ie1, ie2))

        iso_error1.append(ie1)
        iso_error2.append(ie2)

    pairing_error = list()
    if light:
        # Compare RF distance between the heavy/light tree.
        for hf, lf in zip(heavy, light):
            assert(hf.name == lf.name)
            assert(set([n.name for n in hf.forest[0].tree.traverse() if n.frequency > 0]) == set([n.name for n in lf.forest[0].tree.traverse() if n.frequency > 0]))
            # RF distance between heavy and light chain trees:
            RF_dist = hf.forest[0].tree.robinson_foulds(lf.forest[0].tree, unrooted_trees=True)[0]
            pairing_error.append(RF_dist)
        df = pd.DataFrame({'method': methods, 'N_taxa': n_taxa, 'iso_error1': iso_error1, 'iso_error2': iso_error2, 'pairing_error': pairing_error},
                          columns=('method', 'N_taxa', 'iso_error1', 'iso_error2', 'pairing_error'))
    else:
        df = pd.DataFrame({'method': methods, 'N_taxa': n_taxa, 'iso_error1': iso_error1, 'iso_error2': iso_error2},
                          columns=('method', 'N_taxa', 'iso_error1', 'iso_error2'))
    df.to_csv(outbase+'.tsv', sep='\t', index=False)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Validate results of inference on experimental data.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('forests', type=str, nargs='*', help='.p files containing heavy/light chain forests from each inference method.')
    parser.add_argument('--outbase', type=str, required=True, help='output file base name')
    args = parser.parse_args()

    heavy = list()
    light = list()
    for f in args.forests:
        if 'heavy/' in f:
            heavy.append(f)
        elif 'light/' in f:
            light.append(f)
        else:
            raise Exception('No chain tag found in path.')
    assert(len(heavy) > 0 or len(light) == len(heavy))

    heavy_inference = list()
    light_inference = list()
    if len(light) > 0:
        for h, l in zip(heavy, light):
            with open(h, 'rb') as f:
                forest = pickle.load(f)
                heavy_inference.append(forest)
            with open(l, 'rb') as f:
                forest = pickle.load(f)
                light_inference.append(forest)
    else:
        for h in heavy:
            with open(h, 'rb') as f:
                forest = pickle.load(f)
                heavy_inference.append(forest)

    validate(heavy_inference, light_inference, args.outbase)
    print('Done')  # Print something to the log to make the scons wait_func run smoothly


if __name__ == '__main__':
    main()
