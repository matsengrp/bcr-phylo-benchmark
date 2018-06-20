#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''
Parse the output ancestral state reconstruction from FastML together with the topology
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
    except:
        raise TreeFileParsingError('Could not read the input tree. Is this really newick format?')

    tree = reroot_tree(tree, pattern=args.naive)

    counts = {l.split(',')[0]:int(l.split(',')[1]) for l in open(args.counts)}
    tree.add_feature('frequency', 0)       # Placeholder will be deleted when rerooting
    tree.add_feature('sequence', 'DUMMY')  # Placeholder will be deleted when rerooting
    tree = map_asr_to_tree(args.asr_seq, tree, args.naive, counts)

    tree.dist = 0  # No branch above root
    for node in tree.iter_descendants():
        node.dist = hamming_distance(node.sequence, node.up.sequence)

    fastml_tree = CollapsedTree(tree=tree, name=args.name)
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
    fastml_tree.render(args.outbase + '.svg', colormap=colormap)
    fastml_forest = CollapsedForest(forest=[fastml_tree], name=args.name)
    # Dump tree as newick:
    fastml_forest.write_random_tree(args.outbase+'.tree')
    print('number of trees with integer branch lengths:', fastml_forest.n_trees)

    with open(args.outbase + '.p', 'wb') as f:
        pickle.dump(fastml_forest, f)

    print('Done parsing FastML tree')


def map_asr_to_tree(asr_seq, tree, naiveID, counts):
    '''Takes a FastML fasta header and returns the matching ete3 tree node.'''
    for record in SeqIO.parse(asr_seq, "fasta"):
        name = record.id.strip()
        try:
            node = tree.search_nodes(name=name)[0]
        except IndexError:  # Possibly the stupidest hack ever, but stupid ete3 keeps deleting the root upon rerooting!!!
            node = tree.search_nodes(name='')[0]
            node.name = name

        if name in counts:
            frequency = counts[name]
        else:
            frequency = 0
        # Add the features:
        assert(node)
        node.add_feature('frequency', frequency)
        node.add_feature('sequence', str(record.seq))
        #print(node.sequence)

    return tree


def which(executable):
    for path in os.environ["PATH"].split(os.pathsep):
        if os.path.exists(os.path.join(path, executable)):
            return os.path.realpath(os.path.join(path, executable))
    return None


def dedup_fasta(args):
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio.Alphabet import generic_dna
    from Bio import AlignIO
    from Bio.Phylo.TreeConstruction import MultipleSeqAlignment

    aln = AlignIO.read(args.infile, 'fasta')

    seqs_unique_counts = {}
    for seq in aln:
        seq.id = seq.id.strip()
        # if id is just an integer, assume it represents count of that sequence
        if seq.id == args.naive:
            naive_seq = str(seq.seq)  # We want to keep the identity of the naive sequence
        elif seq.id.isdigit():
            seqs_unique_counts[str(seq.seq)] = int(seq.id)
        elif str(seq.seq) not in seqs_unique_counts:
            seqs_unique_counts[str(seq.seq)] = 1
        else:
            seqs_unique_counts[str(seq.seq)] += 1

    with open(args.outfile, 'w') as fh_out:
        print('>' + args.naive, file=fh_out)
        print(naive_seq, file=fh_out)
        for i, seq in enumerate(seqs_unique_counts):
            record = SeqRecord(Seq(seq, generic_dna), id=str(i+1)+'_'+str(seqs_unique_counts[seq]))
            print('>' + str(record.id), file=fh_out)
            print(str(record.seq), file=fh_out)

    print('Done deduplicating input fasta file.')


def reroot(args):
    from ete3 import Tree, NodeStyle, TreeStyle, TextFace, add_face_to_node

    try:
        tree = Tree(args.tree)
    except:
        raise TreeFileParsingError('Could not read the input tree. Is this really newick format?')

    rerotted_tree = reroot_tree(tree, pattern=args.pattern, outgroup=args.outgroup)
    rerotted_tree.write(outfile=args.reroot_tree)

    print('Done rerooting input tree.')


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
        node.add_child(tree.get_children()[0])
        tree.dist = node.dist
        node.dist = 0
        tree = node
    return tree



def main():
    import argparse

    parser = argparse.ArgumentParser(description='Tools for running ASR with FastML.')
    subparsers = parser.add_subparsers(help='Which program to run')

    # Parser for ASR_parser subprogram:
    parser_asr = subparsers.add_parser('ASR_parser',
                                       help='Reroot a tree based on node containing a keyword.',
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_asr.add_argument('--tree', required=True, metavar='NEWICK TREE', help='Input tree used for topology.')
    parser_asr.add_argument('--name', required=True, help='Name of tree.')
    parser_asr.add_argument('--colormap', required=False, help='Colormap for ETE3.')
    parser_asr.add_argument('--idmap', required=False, help='Id mapping from simulation to Phylip file sequence names.')
    parser_asr.add_argument('--counts', required=True, metavar='ALLELE_FREQUENCY', help='File containing allele frequencies (sequence counts) in the format: "SeqID,Nobs"')
    parser_asr.add_argument('--asr_seq', required=True, help='Input ancestral sequences.')
    parser_asr.add_argument('--outbase', required=True, metavar='FILENAME', help='Filename for the output ASR tree.')
    parser_asr.add_argument('--naive', type=str, default='naive', help='naive sequence id')
    parser_asr.set_defaults(func=ASR_parser)

    # Parser for dedup_fasta subprogram:
    parser_dedup = subparsers.add_parser('dedup_fasta',
                                         help='Deduplicate a fasta file.',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_dedup.add_argument('--infile', type=str, help='fasta file with any integer ids indicating frequency', required=True)
    parser_dedup.add_argument('--outfile', type=str, help='Output filename.', required=True)
    parser_dedup.add_argument('--naive', type=str, default='naive', help='naive sequence id')
    parser_dedup.set_defaults(func=dedup_fasta)

    # Parser for reroot subprogram:
    parser_reroot = subparsers.add_parser('reroot',
                                          description='Reroot a tree based on node containing a keyword.',
                                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_reroot.add_argument('--tree', required=True, metavar='NEWICK TREE', help='Input tree to root, in newick format.')
    parser_reroot.add_argument('--reroot_tree', required=True, metavar='FILENAME', help='Filename for the output rerooted tree.')
    parser_reroot.add_argument('--pattern', metavar='REGEX PATTERN', required=True, help="Pattern to search for the node to root on.")
    parser_reroot.add_argument('--outgroup', required=False, action='store_true', default=False, help="Set as outgroup instead of tree root.")
    parser_reroot.set_defaults(func=reroot)


    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
