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
import os, sys

global ISO_TYPE_ORDER
ISO_TYPE_ORDER = [set(['IgM', 'IgD']), set(['IgG', 'IgGA', 'IgGb']), set(['IgE']), set(['IgA'])]
global ALL_ISO_TYPE
ALL_ISO_TYPE = set(['IgM', 'IgD', 'IgG', 'IgGA', 'IgGb', 'IgE', 'IgA'])

def map_meta(args):
    # Read trees:
    tree_dict = dict()
    for forest_file in args.forest_files:
        with open(forest_file, 'rb') as f:
            forest = pickle.load(f)
            tree_dict[forest_file] = forest

    # Read meta info:
    with open(args.meta, 'rb') as f:
        seq_info_dict = pickle.load(f)
    # Read idmap:
    with open(args.idmap, 'rb') as fh:
            id_map = pickle.load(fh)
    # Map meta information:
    for fnam, forest in tree_dict.items():
        for tree in forest.forest:
            for node in tree.tree.traverse():
                if node.frequency > 0:
                    assert(len(id_map[node.name]) > 0)
                    abundance = 0
                    iso_set = set()
                    chain = None
                    for name in id_map[node.name]:
                        meta = seq_info_dict[name]
                        abundance += meta['abundance']
                        iso_set |= set(meta['iso_set'])
                        assert(iso_set & ALL_ISO_TYPE == iso_set)  # All isotypes must be in the known set
                        if chain is None:
                            chain = meta['chain']
                        else:
                            assert(meta['chain'] == chain)
                    # IgM at the root:
                    if node.up is None:
                        iso_set |= set(['IgM'])
                    node.frequency += abundance - 1  # No double counting if seen just once
                    node.add_feature('isotype', iso_set)
                    node.add_feature('chain', chain)
                    node.name = name[:-6]  # trim off _heavy or _light from the name
        forest.forest[0].render(fnam[:-2] + '_meta.svg', isolabel=True)
        with open(fnam[:-2]+'_meta'+fnam[-2:], 'wb') as f:
            pickle.dump(forest, f)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Add meta information e.g. abunddance, isotype and pairing, to a tree.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--meta', type=str, help='Path to .p dict containing meta information.')
    parser.add_argument('--idmap', type=str, help='Path to .p dict containing ID map.')
    parser.add_argument('--forest_files', type=str, nargs='*', help='Paths to .p tree files')
    args = parser.parse_args()
    map_meta(args)

    print('Done')  # Print something to the log to make the scons wait_func run smoothly


if __name__ == '__main__':
    main()
