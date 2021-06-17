#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from Bio.Seq import Seq
from Bio.Seq import translate as bio_translate
from Bio.Alphabet import generic_dna
from ete3 import TreeNode, NodeStyle, TreeStyle, TextFace, CircleFace, PieChartFace, faces, SVG_COLORS
import scipy
import numpy as np
import random
import math

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

# ----------------------------------------------------------------------------------------
def local_translate(seq):
    if len(seq) % 3 != 0:
        seq += 'N' * (3 - (len(seq) % 3))
    return bio_translate(seq)

# ----------------------------------------------------------------------------------------
def replace_codon_in_aa_seq(new_nuc_seq, old_aa_seq, inuc):  # <inuc>: single nucleotide position that was mutated from old nuc seq (which corresponds to old_aa_seq) to new_nuc_seq
    istart = 3 * int(math.floor(inuc / 3.))  # nucleotide position of start of mutated codon
    new_codon = local_translate(new_nuc_seq[istart : istart + 3])
    return old_aa_seq[:inuc / 3] + new_codon + old_aa_seq[inuc / 3 + 1:]  # would be nice to check for synonymity and not do any translation unless we need to

# ----------------------------------------------------------------------------------------
class TranslatedSeq(object):
    # ----------------------------------------------------------------------------------------
    def __init__(self, args, nuc_seq, aa_seq=None):
        self.nuc = nuc_seq
        self.aa = local_translate(nuc_seq) if aa_seq is None else aa_seq
        if args.paratope_positions == 'all':
            self.cdrps_aa = None
            self.cdrps_nuc = None
        elif args.paratope_positions == 'cdrs':
            self.cdrps_aa = list(range(0, int((1./3) * len(self.aa)), 1))  # cdr/paratope positions
            self.cdrps_nuc = [j for i in self.cdrps_aa for j in (3*i, 3*i+1, 3*i+2)]
            # from selection_utils import color
            # print self.aa
            # print ''.join([color('red' if i in self.cdrps_aa else None, a) for i, a in enumerate(self.aa)])
            # print self.nuc
            # print ''.join([color('red' if i in self.cdrps_nuc else None, n) for i, n in enumerate(self.nuc)])
            # sys.exit()
        else:
            assert False
    # ----------------------------------------------------------------------------------------
    def dseq(self, stype):  # return the sequence to use for distance calculations, i.e. only including paratope positions
        if self.cdrps_aa is None:
            return self.aa if stype == 'aa' else self.nuc
        else:
            if stype == 'aa':
                return ''.join(self.aa[i] for i in self.cdrps_aa)  # it might be faster to cache this, but then you'd have to deal with updating it when something mutates, so maybe not
            else:
                return ''.join(self.nuc[i] for i in self.cdrps_nuc)

# ----------------------------------------------------------------------------------------
def has_stop_aa(seq):
    return '*' in seq

# # ----------------------------------------------------------------------------------------
# def has_stop_nuc(seq):  # huh, turns out we don't need this anywhere, but don't feel like deleting, since it makes more clear that the other fcn needs to be passed an aa sequence
#     return has_stop_aa(local_translate(seq))

# ----------------------------------------------------------------------------------------
class CollapsedTree():
    '''
    Collapses an ete3 tree into a genotype collapsed tree based on hamming distance between node seqeunces.
    '''
    def __init__(self, tree, name, meta=None, collapse_syn=False, allow_repeats=False, add_selection_metrics=False):
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

        # TODO remove this, since it happens to the input tree at the end of MutationModel.simulate()
        # Remove unobserved internal unifurcations:
        for node in self.tree.iter_descendants():
            parent = node.up
            if node.frequency == 0 and len(node.children) == 1:
                assert False
                node.delete(prevent_nondicotomic=False)
                node.children[0].dist = hamming_distance(node.children[0].nuc_seq, parent.nuc_seq)

        # Collapse synonymous reads:
        if collapse_syn is True:
            print('Collapsing synonymous reads')
            tree.dist = 0  # No branch above root
            for node in tree.iter_descendants():
                node.dist = hamming_distance(node.aa_seq, node.up.aa_seq)  # NOTE the .dist feature is previously set to be nucleotide distance (but here's it's set to aa distance) THIS IS REALLY DUMB

        # iterate over the tree below root and collapse edges of zero length
        # if the node is a leaf and its parent has nonzero frequency we combine taxa names to a set (this acommodates bootstrap samples that result in repeated genotypes)
        observed_genotypes = set((node.name for node in self.tree.traverse() if node.frequency > 0))
        observed_genotypes.add(self.tree.name)
        for node in self.tree.get_descendants(strategy='postorder'):
            if node.dist != 0:
                continue
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

        n_repeated_seqs = sum(node.frequency > 0 for node in self.tree.traverse()) - len(set([node.nuc_seq for node in self.tree.traverse() if node.frequency > 0]))
        if not allow_repeats and n_repeated_seqs:
            raise RuntimeError('found %d repeated sequences when collapsing tree' % n_repeated_seqs)

        # a custom ladderize accounting for abundance and sequence to break ties in abundance
        for node in self.tree.traverse(strategy='postorder'):
            # add a partition feature and compute it recursively up the tree
            node.add_feature('partition', node.frequency + sum(node2.partition for node2 in node.children))
            # sort children of this node based on partion and sequence
            node.children.sort(key=lambda node: (node.partition, node.nuc_seq))

        if add_selection_metrics:
            self.add_selection_metrics()

    # ----------------------------------------------------------------------------------------
    def __str__(self):
        '''Return a string representation for printing.'''
        return 'tree:\n' + str(self.tree)

    # ----------------------------------------------------------------------------------------
    def add_selection_metrics():
        # Add some usefull annotations, including a metric for selection,
        # inspired/adapted from the LONR score: https://academic.oup.com/nar/article/44/5/e46/2464514
        for node in tree.iter_descendants():
            node.add_feature('NS_dist', hamming_distance(node.aa_seq, node.up.aa_seq))
            if node.is_leaf():
                dist2tip = 0
            else:
                dist2tip = min([hamming_distance(node.nuc_seq, l.nuc_seq) for l in node.iter_descendants() if l.is_leaf()])
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

        # Make a Z-score LONR based on synonymous mutations
        try:
            LONR_syn = np.array([node.LONR for node in tree.iter_descendants() if hasattr(node, 'LONR') and node.NS_dist == 0])
            LONR_syn_mean = np.mean(LONR_syn)
            LONR_syn_std = np.std(LONR_syn)
            for node in tree.iter_descendants():
                if hasattr(node, 'LONR'):
                    node.add_feature('LONR_Zscore', (node.LONR - LONR_syn_mean) / LONR_syn_std)
        except:
            pass

    # ----------------------------------------------------------------------------------------
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
                if set(node.nuc_seq.upper()) == set('ACGT'):  # Don't know what this do, try and delete
                    nonsyn = hamming_distance(node.aa_seq, node.up.aa_seq)
                    if has_stop_aa(node.aa_seq):
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
                aln.append(SeqRecord(Seq(str(node.nuc_seq), generic_dna), id=node.name, description='abundance={}'.format(node.frequency)))
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
            list1 = sorted((node.nuc_seq, node.frequency, node.up.nuc_seq if node.up is not None else None) for node in self.tree.traverse())
            list2 = sorted((node.nuc_seq, node.frequency, node.up.nuc_seq if node.up is not None else None) for node in tree2.tree.traverse())
            return list1 == list2
        elif method == 'MRCA':
            # matrix of hamming distance of common ancestors of taxa
            # takes a true and inferred tree as CollapsedTree objects
            taxa = [node.nuc_seq for node in self.tree.traverse() if node.frequency]
            n_taxa = len(taxa)
            d = scipy.zeros(shape=(n_taxa, n_taxa))
            sum_sites = scipy.zeros(shape=(n_taxa, n_taxa))
            for i in range(n_taxa):
                nodei_true = self.tree.iter_search_nodes(sequence=taxa[i]).next()
                nodei      =      tree2.tree.iter_search_nodes(sequence=taxa[i]).next()
                for j in range(i + 1, n_taxa):
                    nodej_true = self.tree.iter_search_nodes(sequence=taxa[j]).next()
                    nodej      =      tree2.tree.iter_search_nodes(sequence=taxa[j]).next()
                    MRCA_true = self.tree.get_common_ancestor((nodei_true, nodej_true)).nuc_seq
                    MRCA =           tree2.tree.get_common_ancestor((nodei, nodej)).nuc_seq
                    d[i, j] = hamming_distance(MRCA_true, MRCA)
                    sum_sites[i, j] = len(MRCA_true)
            return d.sum() / sum_sites.sum()
        elif method == 'RF':
            tree1_copy = self.tree.copy(method='deepcopy')
            tree2_copy = tree2.tree.copy(method='deepcopy')
            for treex in (tree1_copy, tree2_copy):
                for node in list(treex.traverse()):
                    if node.frequency > 0:
                        # child = init_node(self.args, node.nuc_seq, node.time, node)  # too hard to import this
                        child = TreeNode()
                        child.add_feature('sequence', node.nuc_seq)
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
