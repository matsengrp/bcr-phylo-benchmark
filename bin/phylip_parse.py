#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Given an outputfile from one of the PHYLIP tools - `dnaml` or `dnapars` - produce an alignment (including
ancestral sequences), a newick tree (with matching internal node lables), and an svg rendering of said tree.
"""
from __future__ import print_function
from ete3 import Tree
import re, random
from collections import defaultdict
from Bio.Data.IUPACData import ambiguous_dna_values
from GCutils import hamming_distance, CollapsedTree, CollapsedForest, local_translate


# iterate over recognized sections in the phylip output file.
def sections(fh):
    patterns = {
        "parents": "\s+between\s+and\s+length",
        ("sequences", 'dnaml'): "\s*node\s+reconstructed\s+sequence",
        ('sequences', 'dnapars'): "from\s+to\s+any steps"}
    patterns = {k: re.compile(v, re.IGNORECASE) for (k, v) in patterns.items()}
    for line in fh:
        for k, pat in patterns.items():
            if pat.match(line):
                yield k
                break


# iterate over entries in the distance section
def iter_edges(fh):
    #  152          >naive2           0.01208     (     zero,     0.02525) **
    pat = re.compile("\s*(?P<parent>\w+)\s+(?P<child>[\w>_.-]+)\s+(?P<distance>\d+\.\d+)")
    # drop the header underline
    fh.next()
    matches = 0
    for line in fh:
        m = pat.match(line)
        if m:
            matches += 1
            yield (m.group("child"), m.group("parent"))
        # We only want to break at the very end of the block of matches; dnaml has an extra blank line between
        # header and rows that dnapars doesn't
        elif not matches:
            continue
        else:
            break


# iterate over entries in the sequences section
def parse_seqdict(fh, mode='dnaml'):
    #  152        sssssssssG AGGTGCAGCT GTTGGAGTCT GGGGGAGGCT TGGTACAGCC TGGGGGGTCC
    seqs = defaultdict(str)
    if mode == 'dnaml':
        patterns = re.compile("^\s*(?P<id>[a-zA-Z0-9>_.-]*)\s+(?P<seq>[a-zA-Z \-]+)")
    elif mode == 'dnapars':
        patterns = re.compile("^\s*\S+\s+(?P<id>[a-zA-Z0-9>_.-]*)\s+(yes\s+|no\s+|maybe\s+)?(?P<seq>[a-zA-Z \-]+)")
    else:
        raise ValueError('invalid mode '+mode)
    fh.next()
    for line in fh:
        m = patterns.match(line)
        if m and m.group("id") is not '':
            last_blank = False
            seqs[m.group("id")] += m.group("seq").replace(" ", "").upper()
        elif line.rstrip() == '':
            if last_blank:
                break
            else:
                last_blank = True
                continue
        else:
            break
    return seqs


# parse the dnaml output file and return data structures containing a
# list biopython.SeqRecords and a dict containing adjacency
# relationships and distances between nodes.
def parse_outfile(outfile, countfile=None, naive='naive'):
    '''parse phylip outfile'''
    if countfile is not None:
        counts = {l.split(',')[0]:int(l.split(',')[1]) for l in open(countfile)}
    # No count, just make an empty count dictionary:
    else:
        counts = None
    trees = []
    # Ugg... for compilation need to let python know that these will definely both be defined :-/
    sequences, parents = {}, {}
    with open(outfile, 'rU') as fh:
        for sect in sections(fh):
            if sect == 'parents':
                parents = {child:parent for child, parent in iter_edges(fh)}
            elif sect[0] == 'sequences':
                sequences = parse_seqdict(fh, sect[1])
                # sanity check;  a valid tree should have exactly one node that is parentless
                if not len(parents) == len(sequences) - 1:
                    raise RuntimeError('invalid results attempting to parse {}: there are {} parentless sequences'.format(outfile, len(sequences) - len(parents)))
                trees.append(build_tree(sequences, parents, counts, naive))
            else:
                raise RuntimeError("unrecognized phylip section = {}".format(sect))
    return trees


def disambiguate(tree):
    '''make random choices for ambiguous bases, respecting tree inheritance'''
    sequence_length = len(tree.nuc_seq)
    for node in tree.traverse():
        for site in range(sequence_length):
            base = node.nuc_seq[site]
            if base not in 'ACGT':
                new_base = random.choice(ambiguous_dna_values[base])
                for node2 in node.traverse(is_leaf_fn=lambda n: False if base in [n2.nuc_seq[site] for n2 in n.children] else True):
                    if node2.nuc_seq[site] == base:
                        node2.nuc_seq = node2.nuc_seq[:site] + new_base + node2.nuc_seq[(site+1):]
    return tree


# build a tree from a set of sequences and an adjacency dict.
def build_tree(sequences, parents, counts=None, naive='naive'):
    # build an ete tree
    # first a dictionary of disconnected nodes
    nodes = {}
    for name in sequences:
        node = Tree()
        node.name = name
        node.add_feature('nuc_seq', sequences[node.name])
        node.add_feature('aa_seq', local_translate(sequences[node.name]))
        if counts is not None and node.name in counts:
            node.add_feature('frequency', counts[node.name])
        else:
            node.add_feature('frequency', 0)
        nodes[name] = node
    for name in sequences:
        if name in parents:
            nodes[parents[name]].add_child(nodes[name])
        else:
            tree = nodes[name]
    # Reroot on naive:
    if naive is not None:
        naive_id = [n for n in nodes if naive in n][0]
        assert len(nodes[naive_id].children) == 0
        naive_parent = nodes[naive_id].up
        naive_parent.remove_child(nodes[naive_id])
        nodes[naive_id].add_child(naive_parent)
        # remove possible unecessary unifurcation after rerooting
        if len(naive_parent.children) == 1:
            naive_parent.delete(prevent_nondicotomic=False)
            naive_parent.children[0].dist = hamming_distance(naive_parent.children[0].nuc_seq, nodes[naive_id].nuc_seq)
        tree = nodes[naive_id]

    # make random choices for ambiguous bases
    tree = disambiguate(tree)

    # compute branch lengths
    tree.dist = 0  # no branch above root
    for node in tree.iter_descendants():
        node.dist = hamming_distance(node.nuc_seq, node.up.nuc_seq)

    return tree


def main():
    import pickle, argparse, os

    def existing_file(fname):
        """
        Argparse type for an existing file.
        """
        if not os.path.isfile(fname):
            raise ValueError("File not found: " + str(fname))
        return fname
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name', help="Name of the tree.")
    parser.add_argument('phylip_outfile', type=existing_file, help='dnaml or dnapars outfile (verbose output with inferred ancestral sequences, option 5).')
    parser.add_argument('countfile', nargs='?', type=existing_file, help="Count file.")
    parser.add_argument('--outbase', default='collapsed_forest', help="Output file basename.")
    parser.add_argument('--naive', default='naive', help="Naive sequence id.")
    parser.add_argument('--dump_newick', action='store_true', default=False, help='Dump trees in newick format.')
    parser.add_argument('--colormap', required=False, help='Colormap for ETE3.')
    parser.add_argument('--idmap', required=False, help='Id mapping from simulation to Phylip file sequence names.')
    parser.add_argument('--no-plot', action='store_true', default=False, help='don\'t write any plots.')
    args = parser.parse_args()

    # Parse dnaml/dnapars trees into a collapsed trees and pack them into a forest:
    tree_list = parse_outfile(args.phylip_outfile, args.countfile, args.naive)
    pickle.dump(tree_list[0], open(args.outbase+'_first_lineage.p', 'w'))
    trees = [CollapsedTree(tree=tree, name=args.name) for tree in tree_list]
    forest_obj = CollapsedForest(forest=trees, name=args.name)
    pickle.dump(forest_obj, open(args.outbase+'.p', 'w'))
    if args.dump_newick:
        if forest_obj.n_trees > 1:
            forest_obj.write_trees(args.outbase)
        forest_obj.write_random_tree(args.outbase+'.tree')

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
    # Render svg:
    if not args.no_plot:
        trees[0].render(args.outbase + '.svg', colormap=colormap)
    print('Done')

if __name__ == "__main__":
    main()
