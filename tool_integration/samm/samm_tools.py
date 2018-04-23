#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''
Parse the output ancestral state reconstruction from IQ-TREE together with the topology
to create an ete3 tree with the ancestral sequences.
'''

from __future__ import division, print_function
import sys
import os
import re
from warnings import warn
from Bio import SeqIO
from ete3 import Tree, TreeNode, NodeStyle, TreeStyle, TextFace, add_face_to_node, CircleFace, faces, AttrFace
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../bin')
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../tools')
from Bio import AlignIO

class FastaInputError(Exception):
    '''When the fasta file in not reflecting amino acid DNA coding for protein.'''


class TreeFileParsingError(Exception):
    '''When ete3 fails to read the input tree.'''


def tree_rank(args):
    try:
        import cPickle as pickle
    except:
        import pickle
    from GCutils import CollapsedForest, CollapsedTree, hamming_distance
    from samm.samm_rank import likelihood_of_tree_from_shazam

    with open(args.forest, 'rb') as fh:
        forest_obj = pickle.load(fh)
    forest = forest_obj.forest

    tree_loglik_list = list()
    for tree_obj in forest:
        # Prune tips on tree to max 5 nt. distance to its parent node:
        pruned_tree = tree_obj.tree.copy(method='deepcopy')
        for leaf in pruned_tree.iter_leaves():
            if leaf.dist > 5:
                leaf.delete(prevent_nondicotomic=False)
        print('Before pruning:', tree_obj.tree)
        print('After pruning:', pruned_tree)
        tree_loglik = likelihood_of_tree_from_shazam(pruned_tree, mutability_file=args.mutability_file, substitution_file=args.substitution_file)
        # print(tree_loglik)
        tree_obj.meta['tree_loglik'] = tree_loglik
        tree_loglik_list.append((tree_loglik, tree_obj))

    tree_loglik_list = sorted(tree_loglik_list, key=lambda x: x[0], reverse=True)
    print('List of tree log likelihoods:', [t[0] for t in tree_loglik_list])
    sorted_trees = [t[1] for t in tree_loglik_list]
    sorted_forest_obj = CollapsedForest(forest=sorted_trees, name=args.name)

    # Add colors:
    if args.colormap is not None:
        with open(args.colormap, 'rb') as fh:
            colormap = pickle.load(fh)
        with open(args.idmap, 'rb') as fh:
            id_map = pickle.load(fh)
        # Reverse the id_map:
        id_map = {cs:seq_id for seq_id, cell_ids in id_map.items() for cs in cell_ids}
        # Expand the colormap and map to sequence ids:
        colormap_seqid = dict()
        for key, color in colormap.items():
            if isinstance(key, str) and key in id_map:
                colormap_seqid[id_map[key]] = color
            else:
                for cell_id in key:
                    if cell_id in id_map:
                        colormap_seqid[id_map[cell_id]] = color
        colormap = colormap_seqid
    else:
        colormap = None

    # Render best tree:
    sorted_forest_obj.forest[0].render(args.outbase + '_first.svg', colormap=colormap)
    import copy
    sorted_forest_obj_pruned = copy.deepcopy(sorted_forest_obj)
    # Prune tips on tree to max 5 nt. distance to its parent node:
    for leaf in sorted_forest_obj_pruned.forest[0].tree.iter_leaves():
        if leaf.dist > 5:
            leaf.delete(prevent_nondicotomic=False)
    sorted_forest_obj_pruned.forest[0].render(args.outbase + '_first_pruned.svg', colormap=colormap) 

    # Dump tree as newick:
    sorted_forest_obj.write_first_tree(args.outbase+'.tree')
    print('number of trees with integer branch lengths:', sorted_forest_obj.n_trees)

    with open(args.outbase + '.p', 'wb') as f:
        pickle.dump(sorted_forest_obj, f)

    print('Done ranking forest')


def map_asr_to_tree(asr_seq, leaf_seq, tree, naiveID, counts):
    '''Takes a IQ-TREE asr states and returns the matching ete3 tree node.'''

    # Parse input sequences:
    leafs = list(SeqIO.parse(leaf_seq, "phylip"))
    leafs = {r.id: str(r.seq).upper() for r in leafs}

    # Parse the ASR states from IQ-TREE:
    flag = False
    node_seqs = dict()
    with open(asr_seq) as fh:
        for l in fh:
            if not l.startswith('#') and not flag:
                assert(l.startswith('Node'))
                flag = True
            elif flag:
                state = l.strip().split()
                if state[0] not in node_seqs:
                    node_seqs[state[0]] = list()
                node_seqs[state[0]].append(state[2])
                assert(len(node_seqs[state[0]]) == int(state[1]))
    node_seqs = {k: ''.join(v) for k, v in node_seqs.items()}

    s1 = {len(s) for s in node_seqs.values()}
    s2 = {len(s) for s in leafs.values()}
    assert(s1 == s2)

    # Add info to tree:
    for node in tree.traverse():
        if node.name in counts:
            frequency = counts[node.name]
        else:
            frequency = 0

        if node.name in leafs:
            node_seq = leafs[node.name]
        elif node.name in node_seqs:
            node_seq = node_seqs[node.name]
        else:
            raise Exception('Could not find node name in ancestral or input sequences:', node.name)

        node.add_feature('frequency', frequency)
        node.add_feature('sequence', node_seq)

    return tree


def which(executable):
    for path in os.environ["PATH"].split(os.pathsep):
        if os.path.exists(os.path.join(path, executable)):
            return os.path.realpath(os.path.join(path, executable))
    return None


def find_node(tree, pattern):
    regex = re.compile(pattern).search
    nodes =  [node for node in tree.traverse() for m in [regex(node.name)] if m]
    if not nodes:
        warn("Cannot find matching node; looking for name matching '{}'".format(pattern))
        return
    else:
        if len(nodes) > 1:
            warn("multiple nodes found; using first one.\nfound: {}".format([n.name for n in nodes]))
        return nodes[0]


# reroot the tree on node matching regex pattern.
# Usually this is used to root on the naive germline sequence with a name matching '.*naive.*'
def reroot_tree(tree, pattern='.*naive.*', outgroup=False):
    # find all nodes matching pattern
    node = find_node(tree, pattern)
    tree.set_outgroup(node)
    if tree != node and outgroup:
        s = node.get_sisters()  # KBH - want the outgroup with a branch length of (near) zero
        s[0].dist = node.dist * 2
        node.dist = 0.0000001   # KBH - actual zero length branches cause problems
        tree.swap_children()    # KBH - want root to be at the last taxon in the newick file.
    elif tree != node:
        tree.remove_child(node)
        # Notice that an internal node between the outgroup (naive seq.)
        # and the the root is removed because ASR for IQ-TREE is done using
        # the naive as an actual root.
        node.add_child(tree.get_children()[0].get_children()[0])
        node.add_child(tree.get_children()[0].get_children()[1])
        tree.dist = node.dist
        node.dist = 0
        tree = node
    return tree


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Tools for running ASR with IQ-TREE.')
    subparsers = parser.add_subparsers(help='Which program to run')

    # Parser for tree_rank subprogram:
    parser_rank = subparsers.add_parser('tree_rank',
                                        help='Rank the trees of a forest object. Print the best tree to a newick file and pickle the ranked forest.',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_rank.add_argument('--forest', required=True, metavar='FILE', help='Input forest object containing the trees for likelihood ranking.')
    parser_rank.add_argument('--mutability_file', required=True, metavar='FILE', help='S5F formatted mutability file.')
    parser_rank.add_argument('--substitution_file', required=True, metavar='FILE', help='S5F formatted substitution file.')
    parser_rank.add_argument('--name', required=True, help='Name of the forest.')
    parser_rank.add_argument('--colormap', required=False, help='Colormap for ETE3.')
    parser_rank.add_argument('--idmap', required=False, help='Id mapping from simulation to Phylip file sequence names.')
    parser_rank.add_argument('--outbase', required=True, metavar='FILENAME', help='Filename for the output ASR tree.')
    parser_rank.add_argument('--naive', type=str, default='naive', help='naive sequence id')
    parser_rank.set_defaults(func=tree_rank)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
