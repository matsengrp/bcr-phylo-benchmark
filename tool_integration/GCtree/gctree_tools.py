#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''
This module contains classes for simulation and inference for a binary branching process with mutation
in which the tree is collapsed to nodes that count the number of clonal leaves of each type
'''

from __future__ import division, print_function

import scipy, warnings, random, collections, sys
import pandas as pd, os
from scipy.misc import logsumexp
from scipy.optimize import minimize, check_grad, fsolve
from itertools import cycle
from scipy.stats import poisson
from ete3 import TreeNode, NodeStyle, TreeStyle, TextFace, add_face_to_node, CircleFace, PieChartFace, faces, SVG_COLORS
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import generic_dna
from Bio import AlignIO
from Bio.Phylo.TreeConstruction import MultipleSeqAlignment
import matplotlib; matplotlib.use('agg')
from matplotlib import pyplot as plt, ticker
try:
    import cPickle as pickle
except:
    import pickle

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../')
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../bin')
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../tools')

import phylip_parse
#from bin.GCutils import hamming_distance
#from bin.GCutils import CollapsedForest as newCollapsedForest
#from bin.GCutils import CollapsedTree as newCollapsedTree
from GCutils import hamming_distance
from GCutils import CollapsedForest as newCollapsedForest
from GCutils import CollapsedTree as newCollapsedTree



scipy.seterr(all='raise')


class LeavesAndClades():
    '''
    This is a base class for simulating, and computing likelihood for, an infinite type branching
    process with branching probability p, mutation probability q, and we collapse mutant clades off the
    root type and consider just the number of clone leaves, c, and mutant clades, m.
      /\
     /\ ^          (3)
      /\     ==>   / \\
       /\\
        ^
    '''
    def __init__(self, params=None, c=None, m=None):
        '''initialize with branching probability p and mutation probability q, both in the unit interval'''
        if params is not None:
            p, q = params
            if not (0 <= p <= 1 and 0 <= q <= 1):
                raise ValueError('p and q must be in the unit interval')
        self._nparams = 2#len(params)
        self.params = params
        if c is not None or m is not None:
            if not (c >= 0) and (m >= 0) and (c+m > 0):
                raise ValueError('c and m must be nonnegative integers summing greater than zero')
            self.c = c
            self.m = m

    def simulate(self):
        '''simulate the number of clone leaves and mutant clades off a root node'''
        if self.params[0]>=.5:
            warnings.warn('p >= .5 is not subcritical, tree simulations not garanteed to terminate')
        if self.params is None:
            raise ValueError('params must be defined for simulation\n')

        # let's track the tree in breadth first order, listing number of clonal and mutant descendants of each node
        # mutant clades terminate in this view
        cumsum_clones = 0
        len_tree = 0
        self.c = 0
        self.m = 0
        # while termination condition not met
        while cumsum_clones > len_tree - 1:
            if random.random() < self.params[0]:
                mutants = sum(random.random() < self.params[1] for child in range(2))
                clones = 2 - mutants
                self.m += mutants
            else:
                mutants = 0
                clones = 0
                self.c += 1
            cumsum_clones += clones
            len_tree += 1
        assert cumsum_clones == len_tree - 1

    f_hash = {} # <--- class variable for hashing calls to the following function
    def f(self, params):
        '''
        Probability of getting c leaves that are clones of the root and m mutant clades off
        the root lineage, given branching probability p and mutation probability q
        Also returns gradient wrt (p, q)
        Computed by dynamic programming
        '''
        p, q = params
        c, m = self.c, self.m
        if (p, q, c, m) not in LeavesAndClades.f_hash:
            if c==m==0 or (c==0 and m==1):
                f_result = 0
                dfdp_result = 0
                dfdq_result = 0
            elif c==1 and m==0:
                f_result = 1-p
                dfdp_result = -1
                dfdq_result = 0
            elif c==0 and m==2:
                f_result = p*q**2
                dfdp_result = q**2
                dfdq_result = 2*p*q
            else:
                if m >= 1:
                    neighbor = LeavesAndClades(params=params, c=c, m=m-1)
                    neighbor_f, (neighbor_dfdp, neighbor_dfdq) = neighbor.f(params)
                    f_result = 2*p*q*(1-q)*neighbor_f
                    dfdp_result =   2*q*(1-q) * neighbor_f + \
                                  2*p*q*(1-q) * neighbor_dfdp
                    dfdq_result = (2*p - 4*p*q) * neighbor_f + \
                                   2*p*q*(1-q)  * neighbor_dfdq
                else:
                    f_result = 0.
                    dfdp_result = 0.
                    dfdq_result = 0.
                for cx in range(c+1):
                    for mx in range(m+1):
                        if (not (cx==0 and mx==0)) and (not (cx==c and mx==m)):
                            neighbor1 = LeavesAndClades(params=params, c=cx, m=mx)
                            neighbor2 = LeavesAndClades(params=params, c=c-cx, m=m-mx)
                            neighbor1_f, (neighbor1_dfdp, neighbor1_dfdq) = neighbor1.f(params)
                            neighbor2_f, (neighbor2_dfdp, neighbor2_dfdq) = neighbor2.f(params)
                            f_result += p*(1-q)**2*neighbor1_f*neighbor2_f
                            dfdp_result +=   (1-q)**2 * neighbor1_f    * neighbor2_f + \
                                           p*(1-q)**2 * neighbor1_dfdp * neighbor2_f + \
                                           p*(1-q)**2 * neighbor1_f    * neighbor2_dfdp
                            dfdq_result += -2*p*(1-q) * neighbor1_f    * neighbor2_f + \
                                           p*(1-q)**2 * neighbor1_dfdq * neighbor2_f + \
                                           p*(1-q)**2 * neighbor1_f    * neighbor2_dfdq
            LeavesAndClades.f_hash[(p, q, c, m)] = (f_result, scipy.array([dfdp_result, dfdq_result]))
        return LeavesAndClades.f_hash[(p, q, c, m)]


class CollapsedTree(LeavesAndClades):
    '''
    Here's a derived class for a collapsed tree, where we recurse into the mutant clades
          (4)
         / | \\
       (3)(1)(2)
           |   \\
          (2)  (1)
    '''
    def __init__(self, params=None, tree=None, frame=None, collapse_syn=False, allow_repeats=False):
        '''
        For intialization, either params or tree (or both) must be provided
        params: offspring distribution parameters
        tree: ete tree with frequency node feature. If uncollapsed, it will be collapsed
        frame: tranlation frame, with default None, no tranlation attempted
        '''
        LeavesAndClades.__init__(self, params=params)
        if frame is not None and frame not in (1, 2, 3):
            raise RuntimeError('frame must be 1, 2, 3, or None')
        self.frame = frame

        if collapse_syn is True:
            tree.dist = 0  # no branch above root
            for node in tree.iter_descendants():
                aa = Seq(node.sequence[(frame-1):(frame-1+(3*(((len(node.sequence)-(frame-1))//3))))],
                         generic_dna).translate()
                aa_parent = Seq(node.up.sequence[(frame-1):(frame-1+(3*(((len(node.sequence)-(frame-1))//3))))],
                                generic_dna).translate()
                node.dist = hamming_distance(aa, aa_parent)

        if tree is not None:
            self.tree = tree.copy()
            # remove unobserved internal unifurcations
            for node in self.tree.iter_descendants():
                parent = node.up
                if node.frequency == 0 and len(node.children) == 1:
                    node.delete(prevent_nondicotomic=False)
                    node.children[0].dist = hamming_distance(node.children[0].sequence, parent.sequence)

            # iterate over the tree below root and collapse edges of zero length
            # if the node is a leaf and it's parent has nonzero frequency we combine taxa names to a set
            # this acommodates bootstrap samples that result in repeated genotypes
            observed_genotypes = set((leaf.name for leaf in self.tree))
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

            final_observed_genotypes = set([name for node in self.tree.traverse() if node.frequency > 0 or node == self.tree for name in ((node.name,) if isinstance(node.name, str) else node.name)])
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
        else:
            self.tree = tree

    def l(self, params, sign=1):
        '''
        log likelihood of params, conditioned on collapsed tree, and its gradient wrt params
        optional parameter sign must be 1 or -1, with the latter useful for MLE by minimization
        '''
        if self.tree is None:
            raise ValueError('tree data must be defined to compute likelihood')
        if sign not in (-1, 1):
            raise ValueError('sign must be 1 or -1')
        leaves_and_clades_list = [LeavesAndClades(c=node.frequency, m=len(node.children)) for node in self.tree.traverse()]
        if leaves_and_clades_list[0].c == 0 and leaves_and_clades_list[0].m == 1 and leaves_and_clades_list[0].f(params)[0] == 0:
            # if unifurcation not possible under current model, add a psuedocount for the naive
            leaves_and_clades_list[0].c = 1
        # extract vector of function values and gradient components
        f_data = [leaves_and_clades.f(params) for leaves_and_clades in leaves_and_clades_list]
        fs = scipy.array([[x[0]] for x in f_data])
        logf = scipy.log(fs).sum()
        grad_fs = scipy.array([x[1] for x in f_data])
        grad_logf = (grad_fs/fs).sum(axis=0)
        return sign*logf, sign*grad_logf

    def mle(self, **kwargs):
        '''
        Maximum likelihood estimate for params given tree
        updates params if not None
        returns optimization result
        '''
        # random initalization
        x_0 = (random.random(), random.random())
        bounds = ((.01, .99), (.001, .999))
        kwargs['sign'] = -1
        grad_check = check_grad(lambda x: self.l(x, **kwargs)[0], lambda x: self.l(x, **kwargs)[1], (.4, .5))
        if grad_check > 1e-3:
            warnings.warn('gradient mismatches finite difference approximation by {}'.format(grad_check), RuntimeWarning)
        result = minimize(lambda x: self.l(x, **kwargs), x0=x_0, jac=True, method='L-BFGS-B', options={'ftol':1e-10}, bounds=bounds)
        # update params if None and optimization successful
        if not result.success:
            warnings.warn('optimization not sucessful, '+result.message, RuntimeWarning)
        elif self.params is None:
            self.params = result.x.tolist()
        return result

    def simulate(self):
        '''
        simulate a collapsed tree given params
        replaces existing tree data member with simulation result, and returns self
        '''
        if self.params is None:
            raise ValueError('params must be defined for simulation')

        # initiate by running a LeavesAndClades simulation to get the number of clones and mutants
        # in the root node of the collapsed tree
        LeavesAndClades.simulate(self)
        self.tree = TreeNode()
        self.tree.add_feature('frequency', self.c)
        if self.m == 0:
            return self
        for _ in range(self.m):
            # ooooh, recursion
            child = CollapsedTree(params=self.params, frame=self.frame).simulate().tree
            child.dist = 1
            self.tree.add_child(child)

        return self

    def __str__(self):
        '''return a string representation for printing'''
        return 'params = ' + str(self.params)+ '\ntree:\n' + str(self.tree)

    def render(self, outfile, idlabel=False, colormap=None, show_support=False, chain_split=None):
        '''render to image file, filetype inferred from suffix, svg for color images'''
        def my_layout(node):
            circle_color = 'lightgray' if colormap is None or node.name not in colormap else colormap[node.name]
            text_color = 'black'
            if isinstance(circle_color, str):
                C = CircleFace(radius=max(3, 10*scipy.sqrt(node.frequency)), color=circle_color, label={'text':str(node.frequency), 'color':text_color} if node.frequency > 0 else None)
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
        for node in self.tree.traverse():
            nstyle = NodeStyle()
            nstyle['size'] = 0
            if node.up is not None:
                if set(node.sequence.upper()) == set('ACGT'):
                    if chain_split is not None:
                        if self.frame is not None:
                            raise NotImplementedError('frame not implemented with chain_split')
                        leftseq_mutated = hamming_distance(node.sequence[:chain_split], node.up.sequence[:chain_split]) > 0
                        rightseq_mutated = hamming_distance(node.sequence[chain_split:], node.up.sequence[chain_split:]) > 0
                        if leftseq_mutated and rightseq_mutated:
                            nstyle['hz_line_color'] = 'purple'
                            nstyle['hz_line_width'] = 3
                        elif leftseq_mutated:
                            nstyle['hz_line_color'] = 'red'
                            nstyle['hz_line_width'] = 2
                        elif rightseq_mutated:
                            nstyle['hz_line_color'] = 'blue'
                            nstyle['hz_line_width'] = 2
                    if self.frame is not None:
                        aa = Seq(node.sequence[(self.frame-1):(self.frame-1+(3*(((len(node.sequence)-(self.frame-1))//3))))],
                                 generic_dna).translate()
                        aa_parent = Seq(node.up.sequence[(self.frame-1):(self.frame-1+(3*(((len(node.sequence)-(self.frame-1))//3))))],
                                        generic_dna).translate()
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
        ts.show_branch_support = show_support
        self.tree.render(outfile, tree_style=ts)
        # if we labelled seqs, let's also write the alignment out so we have the sequences (including of internal nodes)
        if idlabel:
            aln = MultipleSeqAlignment([])
            for node in self.tree.traverse():
                aln.append(SeqRecord(Seq(str(node.sequence), generic_dna), id=str(node.name), description='abundance={}'.format(node.frequency)))
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

    def get_split(self, node):
        '''return the bipartition resulting from clipping this node's edge above'''
        if node.get_tree_root() != self.tree:
            raise ValueError('node not found')
        if node == self.tree:
            raise ValueError('this node is the root (no split above)')
        parent = node.up
        taxa1 = []
        for node2 in node.traverse():
            if node2.frequency > 0 or node2 == self.tree:
                if isinstance(node2.name, str):
                    taxa1.append(node2.name)
                else:
                    taxa1.extend(node2.name)
        taxa1 = set(taxa1)
        node.detach()
        taxa2 = []
        for node2 in self.tree.traverse():
            if node2.frequency > 0 or node2 == self.tree:
                if isinstance(node2.name, str):
                    taxa2.append(node2.name)
                else:
                    taxa2.extend(node2.name)
        taxa2 = set(taxa2)
        parent.add_child(node)
        assert taxa1.isdisjoint(taxa2)
        assert taxa1.union(taxa2) == set((name for node in self.tree.traverse() if node.frequency > 0 or node == self.tree for name in ((node.name,) if isinstance(node.name, str) else node.name)))
        return tuple(sorted([taxa1, taxa2]))

    @staticmethod
    def split_compatibility(split1, split2):
        diff = split1[0].union(split1[1]) ^ split2[0].union(split2[1])
        if diff:
            raise ValueError('splits do not cover the same taxa\n\ttaxa not in both: {}'.format(diff))
        for partition1 in split1:
            for partition2 in split2:
                if partition1.isdisjoint(partition2):
                    return True
        return False

    def support(self, bootstrap_trees_list, weights=None, compatibility=False):
        '''
        compute support from a list of bootstrap GCtrees
        weights (optional) is needed for weighting parsimony degenerate trees
        compatibility mode counts trees that don't disconfirm the split
        '''
        for node in self.tree.get_descendants():
            split = self.get_split(node)
            support = 0
            compatibility_ = 0
            for i, tree in enumerate(bootstrap_trees_list):
                compatible = True
                supported = False
                for boot_node in tree.tree.get_descendants():
                    boot_split = tree.get_split(boot_node)
                    if compatibility and compatible and not self.split_compatibility(split, boot_split):
                        compatible = False
                    if not compatibility and not supported and boot_split == split:
                        supported = True
                if supported:
                    support += weights[i] if weights is not None else 1
                if compatible:
                    compatibility_ += weights[i] if weights is not None else 1
            node.support = compatibility_ if compatibility else support

        return self



class CollapsedForest(CollapsedTree):
    '''
    simply a set of CollapsedTrees, with the same p and q parameters
          (4)          (3)
         / | \\         / \\
       (3)(1)(2)     (1) (2)
           |   \\  ,          , ...
          (2)  (1)
    '''
    def __init__(self, params=None, n_trees=None, forest=None):
        '''
        in addition to p and q, we need number of trees
        can also intialize with forest, a list of trees, each an instance of CollapsedTree
        '''
        CollapsedTree.__init__(self, params=params)
        if forest is None and params is None:
            raise ValueError('either params or forest (or both) must be provided')
        if forest is not None:
            if len(forest) == 0:
                raise ValueError('passed empty tree list')
            if n_trees is not None and len(forest) != n_trees:
                raise ValueError('n_trees not consistent with forest')
            self.forest = forest
        if n_trees is not None:
            if type(n_trees) is not int or n_trees < 1:
                raise ValueError('number of trees must be at least one')
            self.n_trees = n_trees
        if n_trees is None and forest is not None:
            self.n_trees = len(forest)

    def simulate(self):
        '''
        simulate a forest of collapsed trees given params and number of trees
        replaces existing forest data member with simulation result, and returns self
        '''
        if self.params is None or self.n_trees is None:
            raise ValueError('params and n_trees parameters must be defined for simulation')
        self.forest = [CollapsedTree(self.params).simulate() for x in range(self.n_trees)]
        return self

    def l(self, params, sign=1, empirical_bayes_sum=False):
        '''
        likelihood of params, given forest, and it's gradient wrt params
        optional parameter sign must be 1 or -1, with the latter useful for MLE by minimization
        if optional parameter empirical_bayes_sum is true, we're doing the Vlad sum for estimating params for
        as set of parsimony trees
        '''
        if self.forest is None:
            raise ValueError('forest data must be defined to compute likelihood')
        if sign not in (-1, 1):
            raise ValueError('sign must be 1 or -1')
        # since the l method on the CollapsedTree class returns l and grad_l...
        terms = [tree.l(params) for tree in self.forest]
        ls = scipy.array([term[0] for term in terms])
        grad_ls = scipy.array([term[1] for term in terms])
        if empirical_bayes_sum:
            # we need to find the smallest derivative component for each
            # coordinate, then subtract off to get positive things to logsumexp
            grad_l = []
            for j in range(len(params)):
                i_prime = grad_ls[:,j].argmin()
                grad_l.append(grad_ls[i_prime,j] +
                              scipy.exp(logsumexp(ls - ls[i_prime],
                                                  b=grad_ls[:,j]-grad_ls[i_prime,j]) -
                                        logsumexp(ls - ls[i_prime])))
            return sign*(-scipy.log(len(ls)) + logsumexp(ls)), sign*scipy.array(grad_l)
        else:
            return sign*ls.sum(), sign*grad_ls.sum(axis=0)

    # NOTE: we get mle() method for free by inheritance/polymorphism magic

    def __str__(self):
        '''return a string representation for printing'''
        return 'params = {}, n_trees = {}\n'.format(self.params, self.n_trees) + \
               '\n'.join([str(tree) for tree in self.forest])


def convert(args):
    with open(args.forest, 'rb') as fh:
        forest_obj = pickle.load(fh)
    tree_list = [tree.tree for tree in forest_obj.forest]
    new_trees = [newCollapsedTree(tree=tree, name=args.name) for tree in tree_list]
    new_forest_obj = newCollapsedForest(forest=new_trees, name=args.name)

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
    new_forest_obj.forest[0].render(args.outbase + '_first.svg', colormap=colormap)
    # Dump tree as newick:
    new_forest_obj.write_first_tree(args.outbase+'.tree')
    print('number of trees with integer branch lengths:', new_forest_obj.n_trees)

    with open(args.outbase + '.p', 'wb') as f:
        pickle.dump(new_forest_obj, f)

    print('Done converting forest')


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Tools for converting GCtree tree/forest objects.')
    subparsers = parser.add_subparsers(help='Which program to run')

    # Parser for the convert subprogram:
    parser_rank = subparsers.add_parser('convert',
                                        help='Convert GCtree objects.',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_rank.add_argument('--forest', required=True, metavar='FILE', help='Input forest object from GCtree.')
    parser_rank.add_argument('--name', required=True, help='Name of the forest.')
    parser_rank.add_argument('--colormap', required=False, help='Colormap for ETE3.')
    parser_rank.add_argument('--idmap', required=False, help='Id mapping from simulation to Phylip file sequence names.')
    parser_rank.add_argument('--outbase', required=True, metavar='FILENAME', help='Filename for the output ASR tree.')
    parser_rank.add_argument('--naive', type=str, default='naive', help='naive sequence id')
    parser_rank.set_defaults(func=convert)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
