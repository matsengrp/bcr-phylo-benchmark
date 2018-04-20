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
from Bio import AlignIO

class FastaInputError(Exception):
    '''When the fasta file in not reflecting amino acid DNA coding for protein.'''


class TreeFileParsingError(Exception):
    '''When ete3 fails to read the input tree.'''


def ASR_parser(args):
    try:
        import cPickle as pickle
    except:
        import pickle
    from GCutils import CollapsedForest, CollapsedTree, hamming_distance

    try:
        tree = Tree(args.tree, format=1)
    except Exception as e:
        print(e)
        raise TreeFileParsingError('Could not read the input tree. Is this really newick format?')

    counts = {l.split(',')[0]:int(l.split(',')[1]) for l in open(args.counts)}
    tree.add_feature('frequency', 0)       # Placeholder will be deleted when rerooting
    tree.add_feature('sequence', 'DUMMY')  # Placeholder will be deleted when rerooting
    tree = map_asr_to_tree(args.asr_seq, args.leaf_seq, tree, args.naive, counts)

    # Reroot to make the naive sequence the real root instead of just an outgroup:
    tree = reroot_tree(tree, pattern=args.naive)

    # Recompute branch lengths as hamming distances:
    tree.dist = 0  # No branch above root
    for node in tree.iter_descendants():
        node.dist = hamming_distance(node.sequence, node.up.sequence)

    iqtree_tree = CollapsedTree(tree=tree, name=args.name)
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
    iqtree_tree.render(args.outbase + '.svg', colormap=colormap)
    iqtree_forest = CollapsedForest(forest=[iqtree_tree], name=args.name)
    # Dump tree as newick:
    iqtree_forest.write_random_tree(args.outbase+'.tree')
    print('number of trees with integer branch lengths:', iqtree_forest.n_trees)

    with open(args.outbase + '.p', 'wb') as f:
        pickle.dump(iqtree_forest, f)

    print('Done parsing IQ-TREE tree')


def map_asr_to_tree(asr_seq, leaf_seq, tree, naiveID, counts):
    '''Takes a IQ-TREE asr states and returns the matching ete3 tree node.'''

    DNA_order = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
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
                max_prop = max(map(float, state[3:7]))
                MAP_base = set([DNA_order[i] for i, p in enumerate(map(float, state[3:7])) if p == max_prop])
                if state[2] in MAP_base:
                    node_seqs[state[0]].append(state[2])
                elif state[2] == '-':
                    node_seqs[state[0]].append(MAP_base.pop())
                else:
                    raise Exception('Sequence reconstruction by IQTREE is inconsistent: {}'.format(l))
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

    # Parser for ASR_parser subprogram:
    parser_asr = subparsers.add_parser('ASR_parser',
                                       help='Reroot a tree based on node containing a keyword.',
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_asr.add_argument('--tree', required=True, metavar='NEWICK TREE', help='Input tree used for topology.')
    parser_asr.add_argument('--name', required=True, help='Name of the tree.')
    parser_asr.add_argument('--colormap', required=False, help='Colormap for ETE3.')
    parser_asr.add_argument('--idmap', required=False, help='Id mapping from simulation to Phylip file sequence names.')
    parser_asr.add_argument('--counts', required=True, metavar='ALLELE_FREQUENCY', help='File containing allele frequencies (sequence counts) in the format: "SeqID,Nobs"')
    parser_asr.add_argument('--asr_seq', required=True, help='Input ancestral sequences file.')
    parser_asr.add_argument('--leaf_seq', required=True, help='Phylip file with leaf sequences.')
    parser_asr.add_argument('--outbase', required=True, metavar='FILENAME', help='Filename for the output ASR tree.')
    parser_asr.add_argument('--naive', type=str, default='naive', help='naive sequence id')
    parser_asr.set_defaults(func=ASR_parser)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
