#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna
from ete3 import TreeNode, NodeStyle, TreeStyle, TextFace, CircleFace, PieChartFace, faces, SVG_COLORS
import scipy
import numpy as np
import random
try:
    import cPickle as pickle
except:
    import pickle
try:
    import jellyfish

    def hamming_distance(s1, s2):
        if s1 == s2:
            return 0
        else:
            return jellyfish.hamming_distance(unicode(s1), unicode(s2))
except:
    def hamming_distance(seq1, seq2):
        '''Hamming distance between two sequences of equal length'''
        return sum(x != y for x, y in zip(seq1, seq2))
    print('Couldn\'t find the python module "jellyfish" which is used for fast string comparison. Falling back to pure python function.')

global ISO_TYPE_ORDER
global ISO_TYPE_charORDER
global ISO_SHORT
ISO_TYPE_ORDER = {'IgM': 1, 'IgD': 1, 'IgG': 2, 'IgGA': 2, 'IgGb': 2, 'IgE': 3, 'IgA': 4}
ISO_TYPE_charORDER = {'M': 1, 'D': 2, 'G': 3, 'E': 4, 'A': 5}
ISO_SHORT = {'IgM': 'M', 'IgD': 'D', 'IgG': 'G', 'IgGA': 'G', 'IgGb': 'G', 'IgE': 'E', 'IgA': 'A'}

def translate(seq):
    if len(seq) % 3 != 0:
        seq += 'N' * (3 - (len(seq) % 3))
    return str(Seq(seq[:], generic_dna).translate())

def has_stop(seq):
    if len(seq) % 3 != 0:
        seq += 'N' * (3 - (len(seq) % 3))
    return '*' in str(Seq(seq[:], generic_dna).translate())


class CollapsedTree():
    '''
    Collapsed tree class from GCtree. Collapses an ete3 tree
    into a genotype collapsed tree based on hamming distance between node seqeunces.
    '''
    def __init__(self, tree, name, meta=None, collapse_syn=False, allow_repeats=False):
        '''
        meta: dictionary with key value pairs e.g. the likelihood of a given tree
        tree: ete tree with frequency node feature. If uncollapsed, it will be collapsed.
        '''
        self.tree = tree
        self.name = name
        if meta is None:
            self.meta = dict()
        else:
            assert(type(meta) == dict)
            self.meta = meta

        self.tree = tree.copy()
        tree.dist = 0  # no branch above root
        # Remove unobserved internal unifurcations:
        for node in self.tree.iter_descendants():
            parent = node.up
            if node.frequency == 0 and len(node.children) == 1:
                node.delete(prevent_nondicotomic=False)
                node.children[0].dist = hamming_distance(node.children[0].sequence, parent.sequence)

        # Collapse synonymous reads:
        if collapse_syn is True:
            print('Collapsing synonymous reads')
            tree.dist = 0  # No branch above root
            for node in tree.iter_descendants():
                aa = translate(node.sequence)
                aa_parent = translate(node.up.sequence)
                node.dist = hamming_distance(aa, aa_parent)

        # iterate over the tree below root and collapse edges of zero length
        # if the node is a leaf and it's parent has nonzero frequency we combine taxa names to a set
        # this acommodates bootstrap samples that result in repeated genotypes
        observed_genotypes = set((node.name for node in self.tree.traverse() if node.frequency > 0))
        observed_genotypes.add(self.tree.name)
        for node in self.tree.get_descendants(strategy='postorder'):
            if node.dist == 0:
                node.up.frequency += node.frequency
                node_set = set([node.name]) if isinstance(node.name, str) else set(node.name)
                node_up_set = set([node.up.name]) if isinstance(node.up.name, str) else set(node.up.name)
                if node_up_set < observed_genotypes:
                    if node_set < observed_genotypes:
                        node.up.name = tuple(node_set | node_up_set)
                        if len(node.up.name) == 1:
                            node.up.name = node.up.name[0]
                elif node_set < observed_genotypes:
                    node.up.name = tuple(node_set)
                    if len(node.up.name) == 1:
                        node.up.name = node.up.name[0]
                node.delete(prevent_nondicotomic=False)

        final_observed_genotypes = set([nam for node in self.tree.traverse() if node.frequency > 0 or node == self.tree for nam in ((node.name,) if isinstance(node.name, str) else node.name)])
        if final_observed_genotypes != observed_genotypes:
            raise RuntimeError('observed genotypes don\'t match after collapse\n\tbefore: {}\n\tafter: {}\n\tsymmetric diff: {}'.format(observed_genotypes, final_observed_genotypes, observed_genotypes ^ final_observed_genotypes))
        assert sum(node.frequency for node in tree.traverse()) == sum(node.frequency for node in self.tree.traverse())

        rep_seq = sum(node.frequency > 0 for node in self.tree.traverse()) - len(set([node.sequence for node in self.tree.traverse() if node.frequency > 0]))
        if not allow_repeats and rep_seq:
            raise RuntimeError('Repeated observed sequences in collapsed tree. {} sequences were found repeated.'.format(rep_seq))
        elif allow_repeats and rep_seq:
            rep_seq = sum(node.frequency > 0 for node in self.tree.traverse()) - len(set([node.sequence for node in self.tree.traverse() if node.frequency > 0]))
            print('Repeated observed sequences in collapsed tree. {} sequences were found repeated.'.format(rep_seq))
        # a custom ladderize accounting for abundance and sequence to break ties in abundance
        for node in self.tree.traverse(strategy='postorder'):
            # add a partition feature and compute it recursively up the tree
            node.add_feature('partition', node.frequency + sum(node2.partition for node2 in node.children))
            # sort children of this node based on partion and sequence
            node.children.sort(key=lambda node: (node.partition, node.sequence))


        # Add some usefull annotations, including a metric for selection,
        # inspired/adapted from the LONR score: https://academic.oup.com/nar/article/44/5/e46/2464514
        for node in tree.iter_descendants():
            aa = translate(node.sequence)
            aa_parent = translate(node.up.sequence)
            node.add_feature('NS_dist', hamming_distance(aa, aa_parent))
            if node.is_leaf():
                dist2tip = 0
            else:
                dist2tip = min([hamming_distance(node.sequence, l.sequence) for l in node.iter_descendants() if l.is_leaf()])
            node.add_feature('dist2tip', dist2tip)
            if hasattr(node, 'Kd'):
                node.add_feature('delta_Kd', (node.up.Kd - node.Kd))

            parent = node.up
            N_children = len(parent.get_children())
            node_N_leaves = sum([l.frequency for l in node.traverse()])
            parent_N_leaves = sum([l.frequency for l in parent.iter_descendants()])

            if N_children <= 1:
                continue
            try:
                LONR = np.log(float(node_N_leaves) / (float(parent_N_leaves - node_N_leaves) / float(N_children - 1)))
                node.add_feature('LONR', LONR)
            except:
                pass

        # Make a Z-score LONR based of the synonymous mutations:
        try:
            LONR_syn = np.array([node.LONR for node in tree.iter_descendants() if hasattr(node, 'LONR') and node.NS_dist == 0])
            LONR_syn_mean = np.mean(LONR_syn)
            LONR_syn_std = np.std(LONR_syn)
            for node in tree.iter_descendants():
                if hasattr(node, 'LONR'):
                    node.add_feature('LONR_Zscore', (node.LONR - LONR_syn_mean) / LONR_syn_std)
        except:
            pass

    def __str__(self):
        '''Return a string representation for printing.'''
        return 'tree:\n' + str(self.tree)

    def render(self, outfile, idlabel=False, isolabel=False, colormap=None, chain_split=None):
        '''Render to image file, filetype inferred from suffix, svg for color images'''
        def my_layout(node):
            circle_color = 'lightgray' if colormap is None or node.name not in colormap else colormap[node.name]
            text_color = 'black'
            if isinstance(circle_color, str):
                if isolabel and hasattr(node, 'isotype'):
                    nl = ''.join(sorted(set([ISO_SHORT[iss] for iss in node.isotype]), key=lambda x: ISO_TYPE_charORDER[x]))
                else:
                    nl = str(node.frequency)
                C = CircleFace(radius=max(3, 10*scipy.sqrt(node.frequency)), color=circle_color, label={'text': nl, 'color': text_color} if node.frequency > 0 else None)
                C.rotation = -90
                C.hz_align = 1
                faces.add_face_to_node(C, node, 0)
            else:
                P = PieChartFace([100*x/node.frequency for x in circle_color.values()], 2*10*scipy.sqrt(node.frequency), 2*10*scipy.sqrt(node.frequency), colors=[(color if color != 'None' else 'lightgray') for color in list(circle_color.keys())], line_color=None)
                T = TextFace(' '.join([str(x) for x in list(circle_color.values())]), tight_text=True)
                T.hz_align = 1
                T.rotation = -90
                faces.add_face_to_node(P, node, 0, position='branch-right')
                faces.add_face_to_node(T, node, 1, position='branch-right')
            if idlabel:
                T = TextFace(node.name, tight_text=True, fsize=6)
                T.rotation = -90
                T.hz_align = 1
                faces.add_face_to_node(T, node, 1 if isinstance(circle_color, str) else 2, position='branch-right')
            elif isolabel and hasattr(node, 'isotype') and False:
                iso_name = ''.join(sorted(set([ISO_SHORT[iss] for iss in node.isotype]), key=lambda x: ISO_TYPE_charORDER[x]))
                #T = TextFace(iso_name, tight_text=True, fsize=6)
                #T.rotation = -90
                #T.hz_align = 1
                #faces.add_face_to_node(T, node, 1 if isinstance(circle_color, str) else 2, position='branch-right')
                C = CircleFace(radius=max(3, 10*scipy.sqrt(node.frequency)), color=circle_color, label={'text':iso_name, 'color':text_color} if node.frequency > 0 else None)
                C.rotation = -90
                C.hz_align = 1
                faces.add_face_to_node(C, node, 0)
        for node in self.tree.traverse():
            nstyle = NodeStyle()
            nstyle['size'] = 0
            if node.up is not None:
                if set(node.sequence.upper()) == set('ACGT'):  # Don't know what this do, try and delete
                    aa = translate(node.sequence)
                    aa_parent = translate(node.up.sequence)
                    nonsyn = hamming_distance(aa, aa_parent)
                    if '*' in aa:
                        nstyle['bgcolor'] = 'red'
                    if nonsyn > 0:
                        nstyle['hz_line_color'] = 'black'
                        nstyle['hz_line_width'] = nonsyn
                    else:
                        nstyle['hz_line_type'] = 1
            node.set_style(nstyle)

        ts = TreeStyle()
        ts.show_leaf_name = False
        ts.rotation = 90
        ts.draw_aligned_faces_as_table = False
        ts.allow_face_overlap = True
        ts.layout_fn = my_layout
        ts.show_scale = False
        self.tree.render(outfile, tree_style=ts)
        # If we labelled seqs, let's also write the alignment out so we have the sequences (including of internal nodes):
        if idlabel:
            aln = MultipleSeqAlignment([])
            for node in self.tree.traverse():
                aln.append(SeqRecord(Seq(str(node.sequence), generic_dna), id=node.name, description='abundance={}'.format(node.frequency)))
            AlignIO.write(aln, open(os.path.splitext(outfile)[0] + '.fasta', 'w'), 'fasta')

    def write(self, file_name):
        '''serialize tree to file'''
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    def compare(self, tree2, method='identity'):
        '''compare this tree to the other tree'''
        if method == 'identity':
            # we compare lists of seq, parent, abundance
            # return true if these lists are identical, else false
            list1 = sorted((node.sequence, node.frequency, node.up.sequence if node.up is not None else None) for node in self.tree.traverse())
            list2 = sorted((node.sequence, node.frequency, node.up.sequence if node.up is not None else None) for node in tree2.tree.traverse())
            return list1 == list2
        elif method == 'MRCA':
            # matrix of hamming distance of common ancestors of taxa
            # takes a true and inferred tree as CollapsedTree objects
            taxa = [node.sequence for node in self.tree.traverse() if node.frequency]
            n_taxa = len(taxa)
            d = scipy.zeros(shape=(n_taxa, n_taxa))
            sum_sites = scipy.zeros(shape=(n_taxa, n_taxa))
            for i in range(n_taxa):
                nodei_true = self.tree.iter_search_nodes(sequence=taxa[i]).next()
                nodei      =      tree2.tree.iter_search_nodes(sequence=taxa[i]).next()
                for j in range(i + 1, n_taxa):
                    nodej_true = self.tree.iter_search_nodes(sequence=taxa[j]).next()
                    nodej      =      tree2.tree.iter_search_nodes(sequence=taxa[j]).next()
                    MRCA_true = self.tree.get_common_ancestor((nodei_true, nodej_true)).sequence
                    MRCA =           tree2.tree.get_common_ancestor((nodei, nodej)).sequence
                    d[i, j] = hamming_distance(MRCA_true, MRCA)
                    sum_sites[i, j] = len(MRCA_true)
            return d.sum() / sum_sites.sum()
        elif method == 'RF':
            tree1_copy = self.tree.copy(method='deepcopy')
            tree2_copy = tree2.tree.copy(method='deepcopy')
            for treex in (tree1_copy, tree2_copy):
                for node in list(treex.traverse()):
                    if node.frequency > 0:
                        child = TreeNode()
                        child.add_feature('sequence', node.sequence)
                        node.add_child(child)
            try:
                return tree1_copy.robinson_foulds(tree2_copy, attr_t1='sequence', attr_t2='sequence', unrooted_trees=True)[0]
            except:
                return tree1_copy.robinson_foulds(tree2_copy, attr_t1='sequence', attr_t2='sequence', unrooted_trees=True, allow_dup=True)[0]
        else:
            raise ValueError('invalid distance method: ' + method)


class CollapsedForest():
    '''
    A forest of collapsed trees e.g. to store equally parsimonious trees.
    '''
    def __init__(self, forest, name, n_trees=None):
        self.forest = forest
        self.name = name
        if len(forest) == 0:
            raise ValueError('Passed empty tree list')
        if n_trees is not None and len(forest) != n_trees:
            raise ValueError('n_trees not consistent with forest')
        if n_trees is not None:
            if type(n_trees) is not int or n_trees < 1:
                raise ValueError('number of trees must be at least one')
            self.n_trees = n_trees
        if n_trees is None and forest is not None:
            self.n_trees = len(forest)

    def __str__(self):
        '''return a string representation for printing'''
        return 'n_trees = {}\n'.format(self.n_trees) + '\n'.join([str(tree) for tree in self.forest])

    def write_trees(self, outbase):
        '''
        Write all the trees of the forest in newick format.
        Each tree is named by thee "outbase" and suffixed with _1.tree, _2.tree etc.
        '''
        for i, tree in enumerate(self.forest):
            tree.tree.write(outfile='{}_{}.tree'.format(outbase, i))

    def write_random_tree(self, outname):
        '''Pick a random tree in the forest and write it in newick format.'''
        assert outname[-5:] == '.tree'
        i = random.randint(0, self.n_trees - 1)
        self.forest[i].tree.write(outfile=outname)

    def write_first_tree(self, outname):
        '''Pick the first tree in the forest and write it in newick format.'''
        assert outname[-5:] == '.tree'
        self.forest[0].tree.write(outfile=outname)
