#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''
This module contains classes for simulation and inference for a binary branching process with mutation
in which the tree is collapsed to nodes that count the number of clonal leaves of each type
'''

from __future__ import division, print_function
import random
import pandas as pd
import os
import time
import itertools
import scipy
from scipy import stats
import numpy
from colored_traceback import always
import sys
from ete3 import TreeNode, TreeStyle, NodeStyle, SVG_COLORS
import matplotlib; matplotlib.use('agg')
try:
    import cPickle as pickle
except:
    import pickle

from GCutils import hamming_distance, has_stop_aa, translate, replace_codon_in_aa_seq, CollapsedTree, TranslatedSeq
import selection_utils
from selection_utils import target_distance_fcn

scipy.seterr(all='raise')

# ----------------------------------------------------------------------------------------
# For paired heavy/light, the sequences are stored sequentially in one string. This fcn is for extracting them.
def get_pair_seq(joint_seq, pair_bounds, iseq):
    return joint_seq[pair_bounds[iseq][0] : pair_bounds[iseq][1]]

# ----------------------------------------------------------------------------------------
class MutationModel():
    # ----------------------------------------------------------------------------------------
    def __init__(self, args, mutation_order=True):
        """
        initialized with input files of the S5F format
        @param mutation_order: whether or not to mutate sequences using a context sensitive manner
                               where mutation order matters
        """
        self.lambda_min = 10e-10  # small lambdas are causing problems so make a minimum
        self.mutation_order = mutation_order
        self.translation_cache = {}
        if args.mutability_file is not None and args.substitution_file is not None:
            self.context_model = {}
            with open(args.mutability_file, 'r') as f:
                # Eat header:
                f.readline()
                for line in f:
                    motif, score = line.replace('"', '').split()[:2]
                    self.context_model[motif] = float(score)

            # kmer k
            self.k = None
            with open(args.substitution_file, 'r') as f:
                # Eat header:
                f.readline()
                for line in f:
                    fields = line.replace('"', '').split()
                    motif = fields[0]
                    if self.k is None:
                        self.k = len(motif)
                        assert self.k % 2 == 1
                    else:
                        assert len(motif) == self.k
                    self.context_model[motif] = (self.context_model[motif], {b:float(x) for b, x in zip('ACGT', fields[1:5])})
        else:
            self.context_model = None

    # ----------------------------------------------------------------------------------------
    def add_translation(self, nuc_seq, aa_seq):
        self.translation_cache[nuc_seq] = aa_seq

    # ----------------------------------------------------------------------------------------
    def get_translation(self, nuc_seq):
        if nuc_seq not in self.translation_cache:
            self.translation_cache[nuc_seq] = translate(nuc_seq)
        return self.translation_cache[nuc_seq]

    # ----------------------------------------------------------------------------------------
    def init_node(self, args, nuc_seq, time, parent, target_seqs=None, aa_seq=None):
        node = TreeNode()
        node.add_feature('nuc_seq', nuc_seq)  # NOTE don't use a TranslatedSeq feature since it gets written to pickle files, which then requires importing the class definition
        node.add_feature('aa_seq', aa_seq if aa_seq is not None else self.get_translation(nuc_seq))
        node.add_feature('naive_distance', hamming_distance(nuc_seq, args.naive_tseq.nuc))  # always nucleotide distance
        node.add_feature('time', time)
        node.dist = 0 if parent is None else hamming_distance(nuc_seq, parent.nuc_seq)  # always nucleotide distance (doesn't use add_feature() because it's a builtin ete3 feature)
        node.add_feature('terminated', False)  # set if it's dead (only set if it draws zero children, or if --kill_sampled_intermediates is set and it's sampled at an intermediate time point)
        node.add_feature('intermediate_sampled', False)  # set if it's sampled at an intermediate time point
        node.add_feature('frequency', 0)  # observation frequency, is set to either 1 or 0 in set_observation_frequencies_and_names(), then when the collapsed tree is constructed it can get bigger than 1 when the frequencies of nodes connected by zero-length branches are summed
        if args.selection:
            node.add_feature('lambda_', None)  # set in selection_utils.update_lambda_values() (i.e. is modified every (few) generations)
            node.add_feature('target_distance', target_distance_fcn(args, TranslatedSeq(nuc_seq, node.aa_seq), target_seqs))  # nuc or aa distance, depending on args
            node.add_feature('Kd', selection_utils.calc_kd(node, args))
        return node

    # ----------------------------------------------------------------------------------------
    @staticmethod
    def disambiguate(sequence):
        '''Generator of all possible nt sequences implied by a sequence containing Ns.'''
        # Find the first N nucleotide:
        N_index = sequence.find('N')
        # If there is no N nucleotide, yield the input sequence:
        if N_index == -1:
            yield sequence
        else:
            for n_replace in 'ACGT':
                # ooooh, recursion
                # NOTE: in python3 we could simply use "yield from..." instead of this loop
                for sequence_recurse in MutationModel.disambiguate(sequence[:N_index] + n_replace + sequence[N_index+1:]):
                    yield sequence_recurse

    # ----------------------------------------------------------------------------------------
    def mutability(self, kmer):
        '''
        Returns the mutability of a central base of a kmer, along with nucleotide bias
        averages over N nucleotide identities.
        '''
        if self.context_model is None:
            raise ValueError('kmer mutability only defined for context models')
        if len(kmer) != self.k:
            raise ValueError('kmer of length {} inconsistent with context model kmer length {}'.format(len(kmer), self.k))
        if not all(n in 'ACGTN' for n in kmer):
            raise ValueError('Sequence {} must contain only characters A, C, G, T, or N'.format(kmer))

        mutabilities_to_average, substitutions_to_average = zip(*[self.context_model[x] for x in MutationModel.disambiguate(kmer)])

        average_mutability = scipy.mean(mutabilities_to_average)
        average_substitution = {b:sum(substitution_dict[b] for substitution_dict in substitutions_to_average)/len(substitutions_to_average) for b in 'ACGT'}

        return average_mutability, average_substitution

    # ----------------------------------------------------------------------------------------
    def mutabilities(self, sequence):
        '''Returns the mutability of a sequence at each site, along with nucleotide biases.'''
        # Pad with Ns to allow averaged edge effects:
        sequence = 'N'*(self.k//2) + sequence + 'N'*(self.k//2)
        # Mutabilities of each nucleotide:
        return [self.mutability(sequence[(i-self.k//2):(i+self.k//2+1)]) for i in range(self.k//2, len(sequence) - self.k//2)]

    # ----------------------------------------------------------------------------------------
    def mutate(self, nuc_seq, lambda0, aa_seq=None, debug=False):
        """
        Mutate a sequence, with lamdba0 the baseline mutability. Cannot mutate the same position multiple times.
        If <aa_seq> is set, then we update it and return the aa_seq for the final nucleotide sequence.
        """

        mutabilities = None
        sequence_mutability = 1.
        if self.context_model is not None:
            mutabilities = self.mutabilities(nuc_seq)
            sequence_mutability = sum(mty[0] for mty in mutabilities) / len(nuc_seq)
        lambda_sequence = sequence_mutability * lambda0  # Poisson rate for this sequence (given its relative mutability):

        n_mutations = numpy.random.poisson(lambda_sequence)

        # Introduce mutations (note: we very commonly just return, i.e. if the poisson kicks up zero mutations)
        unmutated_positions = range(len(nuc_seq))
        if debug:
            dbg_str = []
            print('     adding %d mutations:  ' % n_mutations, end='')
            sys.stdout.flush()
        for i in range(n_mutations):
            # Determine the position to mutate from the mutability matrix:
            mutability_p = None
            if self.context_model is not None:
                mutability_p = scipy.array([mutabilities[pos][0] for pos in unmutated_positions])
                mutability_p /= mutability_p.sum()
            mut_pos = scipy.random.choice(unmutated_positions, p=mutability_p)

            # Now draw the target nucleotide using the substitution matrix
            nucs = [n for n in 'ACGT' if n != nuc_seq[mut_pos]]
            substitution_p = None
            if self.context_model is not None:
                substitution_p = [mutabilities[mut_pos][1][n] for n in nucs]
                assert 0 <= abs(sum(substitution_p) - 1.) < 1e-10
            new_nuc = scipy.random.choice(nucs, p=substitution_p)
            if debug:
                dbg_str += ['%s%d%s' % (nuc_seq[mut_pos], mut_pos, new_nuc)]
            nuc_seq = nuc_seq[ : mut_pos] + new_nuc + nuc_seq[mut_pos + 1 :]
            if aa_seq is not None:
                aa_seq = replace_codon_in_aa_seq(nuc_seq, aa_seq, mut_pos)
                self.add_translation(nuc_seq, aa_seq)
            if self.context_model is not None and self.mutation_order:  # If mutation order matters, the mutabilities of the sequence need to be updated
                mutabilities = self.mutabilities(nuc_seq)

        if debug:
            print('  '.join(dbg_str))
        return {'nuc_seq' : nuc_seq, 'aa_seq' : aa_seq, 'n_muts' : n_mutations}

    # ----------------------------------------------------------------------------------------
    # Make a single target sequence with <n_muts> hamming distance from <args.naive_tseq> (nuc or aa distance according to args.metric_for_target_distance)
    def make_target_sequence(self, args, n_max_tries=100):
        assert not has_stop_aa(args.naive_tseq.aa)  # already checked during argument parsing, but we really want to make sure since the loop will blow up if there's a stop to start with
        itry = 0
        while itry < n_max_tries:
            dist = None
            tseq = args.naive_tseq
            while dist is None or dist < args.target_distance:
                mfo = self.mutate(tseq.nuc, args.target_sequence_lambda0, aa_seq=tseq.aa)
                tseq = TranslatedSeq(mfo['nuc_seq'], aa_seq=mfo['aa_seq'])
                dist = target_distance_fcn(args, args.naive_tseq, [tseq])
                if dist == args.target_distance and not has_stop_aa(tseq.aa):  # Stop codon cannot be part of the return
                    return tseq
            itry += 1

        raise RuntimeError('fell through after trying %d times to make a target sequence' % n_max_tries)

    # ----------------------------------------------------------------------------------------
    def get_targets(self, args):
        print('    making %d target sequences' % args.target_count)
        target_seqs = [self.make_target_sequence(args) for i in range(args.target_count)]
        with open(args.outbase + '_targets.fa', 'w') as tfile:
            for itarget, tseq in enumerate(target_seqs):
                tfile.write('>%s\n%s\n' % ('target-%d' % itarget, tseq.nuc))
        assert len(set([len(args.naive_tseq.aa)] + [len(t.aa) for t in target_seqs])) == 1  # targets and naive seq are same length
        assert len(set([args.target_distance] + [target_distance_fcn(args, args.naive_tseq, [t]) for t in target_seqs])) == 1  # all targets are the right distance from the naive
        return target_seqs

    # ----------------------------------------------------------------------------------------
    def choose_leaves_to_sample(self, args, leaves, n_to_sample):
        if args.observe_based_on_affinity:
            raise Exception('needs to be checked')
            probs = [1. / l.Kd for l in leaves]
            total = sum(probs)
            probs = [p / total for p in probs]
            return list(numpy.random.choice(leaves, n_to_sample, p=probs, replace=False))
        else:
            return random.sample(leaves, n_to_sample)

    # ----------------------------------------------------------------------------------------
    def get_target_distance_hist(self, args, leaves):
        if not args.selection:
            return scipy.histogram([0])
        bin_edges = list(numpy.arange(-0.5, int(2 * args.target_distance) + 0.5))  # some sequences manage to wander quite far away from their nearest target without getting killed, so multiply by 2
        hist = scipy.histogram([l.target_distance for l in leaves], bins=bin_edges)  # if <bins> is a list, it defines the bin edges, including the rightmost edge
        return hist

    # ----------------------------------------------------------------------------------------
    def sample_intermediates(self, args, current_time, tree):
        assert len(args.obs_times) == len(args.n_to_sample)
        n_to_sample = args.n_to_sample[args.obs_times.index(current_time)]
        live_nostop_leaves = [l for l in tree.iter_leaves() if not l.terminated and not has_stop_aa(l.aa_seq)]
        if len(live_nostop_leaves) < n_to_sample:
            print('  %s asked to sample more leaves (%d) than are in the tree (%d) at time %d' % (selection_utils.color('red', 'warning'), n_to_sample, len(live_nostop_leaves), current_time))
            n_to_sample = len(live_nostop_leaves)
        inter_sampled_leaves = self.choose_leaves_to_sample(args, live_nostop_leaves, n_to_sample)
        for leaf in inter_sampled_leaves:
            leaf.intermediate_sampled = True
            if args.kill_sampled_intermediates:
                leaf.terminated = True
                self.n_unterminated_leaves -= 1
        self.sampled_tdist_hists[current_time] = self.get_target_distance_hist(args, inter_sampled_leaves)
        print('                  sampled %d (of %d live and stop-free) intermediate leaves (%s) at end of generation %d' % (n_to_sample, len(live_nostop_leaves), 'killing each of them' if args.kill_sampled_intermediates else 'leaving them alive', current_time))

    # ----------------------------------------------------------------------------------------
    def check_termination(self, args, current_time, tree):
        finished = False
        dbgstr, termstr = [], []
        if self.n_unterminated_leaves <= 0:  # if everybody's dead (probably can't actually go less than zero, but not sure)
            termstr += ['    stopping: no unterminated leaves']
            finished = True
        if args.n_final_seqs is not None:
            dbgstr += ['n leaves %d' % self.n_unterminated_leaves]
            if self.n_unterminated_leaves >= args.n_final_seqs:  # if we've got as many sequences as we were asked for
                termstr += ['    --n_final_seqs: breaking with %d >= %d unterminated leaves' % (self.n_unterminated_leaves, args.n_final_seqs)]
                finished = True
        if args.obs_times is not None:
            dbgstr += ['time %d' % current_time]
            if current_time >= max(args.obs_times):  # if we've done as many generations as we were told to
                termstr += ['    --obs_times: breaking at generation %d >= %d' % (current_time, max(args.obs_times))]
                finished = True
        if args.stop_dist is not None and current_time > 0:
            min_tdist = min([l.target_distance for l in tree.iter_leaves()])
            dbgstr += ['min tdist %d' % min_tdist]
            if min_tdist <= args.stop_dist:  # if the leaves have gotten close enough to the target sequences
                termstr += ['    --stop_dist: breaking with min target distance %d <= %d' % (min_tdist, args.stop_dist)]
                finished = True
        return finished, ', '.join(dbgstr), '\n'.join(termstr)

    # ----------------------------------------------------------------------------------------
    def simulate(self, args):
        '''
        Simulate a poisson branching process with mutation introduced
        by the chosen mutation model e.g. motif or uniform.
        Can either simulate under a neutral model without selection,
        or using an affinity muturation inspired model for selection.
        '''

        self.sampled_tdist_hists, self.tdist_hists, self.n_mutated_hists = [None], [None], [None]
        target_seqs = None
        if args.selection:
            target_seqs = self.get_targets(args)

        current_time = 0
        self.n_unterminated_leaves = 1
        tree = self.init_node(args, args.naive_tseq.nuc, 0, None, target_seqs)
        if args.selection:
            self.tdist_hists[0] = self.get_target_distance_hist(args, tree)  # i guess the first entry in the other two just stays None

        if args.debug == 1:
            print('        end of      live     target dist (%s)          kd           termination' % args.metric_for_target_distance)
            print('      generation   leaves      min  mean           min    mean        checks')
        if args.debug > 1:
            print('       gen   live leaves')
            print('             before/after   ileaf   n children     n mutations          Kd   (%s: updated lambdas)' % selection_utils.color('blue', 'x'))
        static_live_leaves, updated_live_leaves = None, None
        while True:
            current_time += 1

            self.sampled_tdist_hists.append(None)
            self.tdist_hists.append(None)
            self.n_mutated_hists.append(None)

            skip_lambda_n = 0  # index keeping track of how many leaves since we last updated all the lambdas
            if static_live_leaves is None:
                static_live_leaves = [l for l in tree.iter_leaves() if not l.terminated]  # NOTE this is out of date as soon as we've added any children in the loop, or killed anybody with no children (but we need it to iterate over, at least as currently set up)
            else:
                static_live_leaves = updated_live_leaves
            updated_live_leaves = [l for l in static_live_leaves]  # but this one, we keep updating (so we don't have to call iter_leaves() so much, which was taking quite a bit of time) (matches self.n_unterminated_leaves)
            random.shuffle(static_live_leaves)
            if args.debug > 1:
                print('      %3d    %3d' % (current_time, len(static_live_leaves)), end='\n' if len(static_live_leaves) == 0 else '')  # NOTE these are live leaves *after* the intermediate sampling above
            for leaf in static_live_leaves:
                if args.selection:
                    lambda_update_dbg_str = ' '
                    if skip_lambda_n == 0:  # time to update the lambdas for every leaf
                        skip_lambda_n = args.skip_update + 1  # reset it so we don't update again until we've done <args.skip_update> more leaves (+ 1 is so that if args.skip_update is 0 we don't skip at all, i.e. args.skip_update is the number of leaves skipped, *not* the number of leaves *between* updates)
                        tree = selection_utils.update_lambda_values(tree, updated_live_leaves, args.A_total, args.B_total, args.Lp)
                        lambda_update_dbg_str = selection_utils.color('blue', 'x')
                    this_lambda = max(leaf.lambda_, self.lambda_min)
                    skip_lambda_n -= 1
                else:
                    this_lambda = args.lambda_

                n_children = numpy.random.poisson(this_lambda)
                if current_time == 1 and len(static_live_leaves) == 1:  # if the first leaf draws zero children, keep trying so we don't have to throw out the whole tree and start over
                    itry = 0
                    while n_children == 0:
                        n_children = numpy.random.poisson(this_lambda)
                        itry += 1
                        if itry > 10 == 0:
                            print('too many tries to get at least one child, giving up on tree')
                            break

                self.n_unterminated_leaves += n_children - 1
                if n_children == 0:  # kill everyone with no children
                    leaf.terminated = True
                    updated_live_leaves.remove(leaf)
                    if len(static_live_leaves) == 1:
                        print('  terminating only leaf in tree because it has no children')

                if args.debug > 1:
                    n_mutation_list, kd_list = [], []
                for _ in range(n_children):
                    if args.naive_seq2 is not None:  # for paired heavy/light we mutate them separately with their own mutation rate
                        mfos = [self.mutate(get_pair_seq(leaf.nuc_seq, args.pair_bounds, iseq), args.lambda0[iseq]) for iseq in range(len(args.lambda0))]  # NOTE doesn't pass or get aa_seq, but only result of that should be that self.init_node() has to calculate it
                        mutated_sequence = ''.join(m['nuc_seq'] for m in mfos)
                    else:
                        mfo = self.mutate(leaf.nuc_seq, args.lambda0[0], aa_seq=leaf.aa_seq)
                        if args.debug > 1:
                            n_mutation_list.append(mfo['n_muts'])
                    child = self.init_node(args, mfo['nuc_seq'], current_time, leaf, target_seqs, aa_seq=mfo['aa_seq'])
                    if args.selection and args.debug > 1:
                        kd_list.append(child.Kd)
                    leaf.add_child(child)
                    updated_live_leaves.append(child)
                    if leaf in updated_live_leaves:  # <leaf> isn't a leaf any more, since now it has children
                        updated_live_leaves.remove(leaf)  # now that it's been updated, it matches self.n_unterminated_leaves
                if args.debug > 1:
                    n_mutation_str_list = [('%d' % n) if n > 0 else '-' for n in n_mutation_list]
                    kd_str_list = ['%.0f' % kd for kd in kd_list]
                    pre_leaf_str = '' if static_live_leaves.index(leaf) == 0 else ('%12s %3d' % ('', len(updated_live_leaves)))
                    print(('      %s      %4d   %3d  %s          %-14s       %-28s') % (pre_leaf_str, static_live_leaves.index(leaf), n_children, lambda_update_dbg_str, ' '.join(n_mutation_str_list), ' '.join(kd_str_list)))

            if args.selection:
                self.tdist_hists[current_time] = self.get_target_distance_hist(args, updated_live_leaves)
                self.n_mutated_hists[current_time] = scipy.histogram([l.naive_distance for l in updated_live_leaves], bins=list(numpy.arange(-0.5, (max(args.obs_times) if args.obs_times is not None else current_time) + 0.5)))  # can't have more than one mutation per generation

            finished, dbgstr, termstr = self.check_termination(args, current_time, tree)

            if args.debug == 1:
                mintd, meantd = '-', '-'
                minkd, meankd = '-', '-'
                if args.selection:
                    tmptdvals = [l.target_distance for l in updated_live_leaves]
                    mintd, meantd = '%2d' % min(tmptdvals), '%3.1f' % numpy.mean(tmptdvals)
                    tmpkdvals = [l.Kd for l in updated_live_leaves if l.Kd != float('inf')]
                    minkd, meankd = [('%5.1f' % v) for v in (min(tmpkdvals), numpy.mean(tmpkdvals))]
                print('        %3d       %5d         %s  %s          %s  %s       %s' % (current_time, len(updated_live_leaves), mintd, meantd, minkd, meankd, dbgstr))

            if finished:
                print(termstr)
                break

            if args.obs_times is not None and len(args.obs_times) > 1 and current_time in args.obs_times:
                self.sample_intermediates(args, current_time, tree)

        # write a histogram of the hamming distances to target at each generation
        if args.selection:
            with open(args.outbase + '_min_aa_target_hdists.p', 'wb') as histfile:
                pickle.dump(self.tdist_hists, histfile)
            with open(args.outbase + '_n_mutated_nuc_hdists.p', 'wb') as histfile:
                pickle.dump(self.n_mutated_hists, histfile)

        stop_leaves = [l for l in tree.iter_leaves() if l.time == current_time and has_stop_aa(l.aa_seq)]
        non_stop_leaves = [l for l in tree.iter_leaves() if l.time == current_time and not has_stop_aa(l.aa_seq)]  # non-stop leaves
        if len(stop_leaves) > 0:
            print('    %d / %d leaves at final time point have stop codons' % (len(stop_leaves), len(stop_leaves) + len(non_stop_leaves)))

        tree.name = 'naive'  # overwritten below if --observe_common_ancestors is set
        potential_names, used_names = None, None
        _, potential_names, used_names = selection_utils.choose_new_uid(potential_names, used_names, initial_length=4, shuffle=True)  # call once (ignoring the returned <uid>) to get the initial length right, and to shuffle them (shuffling is so if we're running multiple events, they have different leaf names, as long as we set the seeds differently)

        if args.obs_times is not None and len(args.obs_times) > 1:  # observe all intermediate sampled nodes
            for node in [l for l in tree.iter_descendants() if l.intermediate_sampled]:
                node.frequency = 1
                uid, potential_names, used_names = selection_utils.choose_new_uid(potential_names, used_names)
                node.name = 'int-' + uid

        observed_leaves = list(non_stop_leaves)  # don't really need a separate list, but it's a little nicer
        if args.n_to_sample is not None and len(observed_leaves) > args.n_to_sample[-1]:  # if there's more leaves than we were asked to sample
            observed_leaves = self.choose_leaves_to_sample(args, observed_leaves, args.n_to_sample[-1])
            print('    sampled %d leaves at final time' % len(observed_leaves))

        for leaf in observed_leaves:
            leaf.frequency = 1
            uid, potential_names, used_names = selection_utils.choose_new_uid(potential_names, used_names)
            leaf.name = 'leaf-' + uid

        if args.selection:
            self.sampled_tdist_hists[current_time] = self.get_target_distance_hist(args, observed_leaves)  # NOTE this doesn't include nodes added from --observe_common_ancestors or --observe_all_ancestors
            if len(self.sampled_tdist_hists) > 0:
                with open(args.outbase + '_sampled_min_aa_target_hdists.p', 'wb') as histfile:
                    pickle.dump(self.sampled_tdist_hists, histfile)

        # prune away lineages that have zero total observation frequency
        n_pruned_lineages, n_pruned_nodes = 0, 0
        start = time.time()
        detached_descendents = set()
        for node in tree.iter_descendants():  # NOTE this is kinda slow, and it might (might!) be faster to propagate the information upward when we set the observed nodes to start with (rather than looping over descendents here)), but it's quite a bit faster than it used to be already so not going to mess around further a.t.m.
            if node in detached_descendents:
                # detached_descendents.remove(node)  # don't need it in there any more, but it isn't any faster to remove it
                continue
            if any(child.frequency > 0 for child in node.traverse()):  # if all children of <node> have zero observation frequency, detach <node> (only difference between traverse() and iter_descendants() seems to be that traverse() includes the node on which you're calling it, while iter_descendants() doesn't)
                continue
            node.detach()
            for child in node.iter_descendants():  # avoid checking <node>'s children
                detached_descendents.add(child)
                n_pruned_nodes += 1
            n_pruned_lineages += 1
        print('    removed %d nodes in %d unobserved lineages (%.1fs)' % (n_pruned_nodes, n_pruned_lineages, time.time()-start))

        # remove unobserved unifurcations
        for node in tree.iter_descendants():
            parent = node.up
            if node.frequency == 0 and len(node.children) == 1:
                node.delete(prevent_nondicotomic=False)  # seems like this should instead use the preserve_branch_length=True option so we don't need the next line, but I don't want to change it
                node.children[0].dist = hamming_distance(node.children[0].nuc_seq, parent.nuc_seq)

        if args.observe_common_ancestors:
            start = time.time()
            n_observed_ancestors = 0
            for ancestor in tree.traverse():
                if ancestor.is_leaf():
                    continue
                ancestor.frequency = 1
                uid, potential_names, used_names = selection_utils.choose_new_uid(potential_names, used_names)
                ancestor.name = 'mrca-' + uid
                n_observed_ancestors += 1
            print('    added %d ancestor nodes (%.1fs)' % (n_observed_ancestors, time.time()-start))  # oh, wait, maybe this doesn't take any real time any more? i thought it used to involve more iteration/traversing

        # neutral collapse will fail if there's backmutations (?) [preserving old comment]
        treename = 'GCsim %s' % ('selection' if args.selection else 'neutral')
        collapsed_tree = CollapsedTree(tree, treename, allow_repeats=args.selection)
        n_collapsed_seqs = sum(node.frequency > 0 for node in collapsed_tree.tree.traverse())
        if n_collapsed_seqs < 2:
            raise RuntimeError('collapsed tree contains only %d observed sequences (we require at least two)' % n_collapsed_seqs)

        tree.ladderize()

        return tree, collapsed_tree


# ----------------------------------------------------------------------------------------
def run_simulation(args):
    mutation_model = MutationModel(args)
    tree, collapsed_tree = None, None
    n_max_tries = 1000
    for itry in range(n_max_tries):  # keep trying if we don't get enough leaves, or if there's backmutation
        try:
            tree, collapsed_tree = mutation_model.simulate(args)
            break
        except RuntimeError as e:
            print('      {} {}\n  trying again'.format(selection_utils.color('red', 'error:'), e))
    if tree is None:
        raise RuntimeError('{} attempts exceeded'.format(n_max_tries))

    # In the case of a sequence pair print them to separate files:
    if args.naive_seq2 is not None:
        fhandles = [open('%s_seq%d.fasta' % (args.outbase, iseq + 1), 'w') for iseq in range(2)]
        for iseq, fh in enumerate(fhandles):
            fh.write('>%s\n%s\n' % (tree.name, get_pair_seq(args.naive_tseq.nuc, args.pair_bounds, iseq)))
        for node in [n for n in tree.iter_descendants() if n.frequency != 0]:  # NOTE doesn't iterate over root node
            for iseq, fh in enumerate(fhandles):
                fh.write('>%s\n%s\n' % (node.name, get_pair_seq(node.nuc_seq, args.pair_bounds, iseq)))
    else:
        with open('%s.fasta' % args.outbase, 'w') as fh:
            fh.write('>%s\n%s\n' % (tree.name, args.naive_tseq.nuc))
            for node in [n for n in tree.iter_descendants() if n.frequency != 0]:  # NOTE doesn't iterate over root node
                fh.write('>%s\n%s\n' % (node.name, node.nuc_seq))

    # write some observable simulation stats
    frequency, distance_from_naive, degree = zip(*[(node.frequency,
                                                    node.naive_distance,
                                                    sum(hamming_distance(node.nuc_seq, node2.nuc_seq) == 1 for node2 in collapsed_tree.tree.traverse() if node2.frequency and node2 is not node))
                                                   for node in collapsed_tree.tree.traverse() if node.frequency])
    stats = pd.DataFrame({'genotype abundance':frequency,
                          'Hamming distance to root genotype':distance_from_naive,
                          'Hamming neighbor genotypes':degree})
    stats.to_csv(args.outbase+'_stats.tsv', sep='\t', index=False)

    print('{} simulated observed sequences'.format(sum(node.frequency for node in collapsed_tree.tree.traverse())))

    # Render the full lineage tree:
    ts = TreeStyle()
    ts.rotation = 90
    ts.show_leaf_name = False
    ts.show_scale = False

    colors = {}
    palette = SVG_COLORS
    palette -= set(['black', 'white', 'gray'])
    palette = itertools.cycle(list(palette))  # <-- Circular iterator

    # Either plot by DNA sequence or amino acid sequence:
    if args.plotAA and args.selection:
        colors[tree.aa_seq] = 'gray'
    else:
        colors[tree.nuc_seq] = 'gray'

    for n in tree.traverse():
        nstyle = NodeStyle()
        nstyle["size"] = 10
        if args.plotAA:
            if n.aa_seq not in colors:
                colors[n.aa_seq] = next(palette)
            nstyle['fgcolor'] = colors[n.aa_seq]
        else:
            if n.nuc_seq not in colors:
                colors[n.nuc_seq] = next(palette)
            nstyle['fgcolor'] = colors[n.nuc_seq]
        n.set_style(nstyle)

    # Render and pickle lineage tree:
    tree.render(args.outbase+'_lineage_tree.svg', tree_style=ts)
    with open(args.outbase+'_lineage_tree.p', 'wb') as f:
        pickle.dump(tree, f)

    # Render collapsed tree,
    # create an id-wise colormap
    # NOTE: node.name can be a set
    if args.plotAA and args.selection:
        colormap = {node.name:colors[node.aa_seq] for node in collapsed_tree.tree.traverse()}
    else:
        colormap = {node.name:colors[node.nuc_seq] for node in collapsed_tree.tree.traverse()}
    collapsed_tree.write(args.outbase+'_collapsed_tree.p')
    collapsed_tree.render(args.outbase+'_collapsed_tree.svg',
                          idlabel=args.idlabel,
                          colormap=colormap)
    # Print colormap to file:
    with open(args.outbase+'_collapsed_tree_colormap.tsv', 'w') as f:
        for name, color in colormap.items():
            f.write((name if isinstance(name, str) else ','.join(name)) + '\t' + color + '\n')
    with open(args.outbase+'_collapsed_tree_colormap.p', 'wb') as f:
        pickle.dump(colormap, f)

    if args.selection:
        # Define a list a suitable colors that are easy to distinguish:
        palette = ['crimson', 'purple', 'hotpink', 'limegreen', 'darkorange', 'darkkhaki', 'brown', 'lightsalmon', 'darkgreen', 'darkseagreen', 'darkslateblue', 'teal', 'olive', 'wheat', 'magenta', 'lightsteelblue', 'plum', 'gold']
        palette = itertools.cycle(list(palette)) # <-- circular iterator
        colors = {i: next(palette) for i in range(int(len(args.naive_tseq.nuc) // 3))}
        # The minimum distance to the target is colored:
        colormap = {node.name:colors[node.target_distance] for node in collapsed_tree.tree.traverse()}
        collapsed_tree.write( args.outbase+'_collapsed_runstat_color_tree.p')
        collapsed_tree.render(args.outbase+'_collapsed_runstat_color_tree.svg',
                              idlabel=args.idlabel,
                              colormap=colormap)
        # Write a file with the selection run stats. These are also plotted:
        with open(args.outbase + '_min_aa_target_hdists.p', 'rb') as fh:
            tdist_hists = pickle.load(fh)
            selection_utils.plot_runstats(tdist_hists, args.outbase, colors)


# ----------------------------------------------------------------------------------------
def main():
    import argparse
    file_path = os.path.realpath(__file__)
    file_dir = os.path.dirname(file_path)

    stop_crits = ['n_final_seqs', 'obs_times', 'stop_dist']
    class MultiplyInheritedFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass
    formatter_class = MultiplyInheritedFormatter
    help_str = '''
    Germinal center simulation. Can simulate in two modes:
      a) Neutral mode. A Galton–Watson process, with mutation probabilities according to a user defined motif model e.g. S5F.
      b) Selection mode. With the same mutation process as in a), but the poisson progeny distribution's lambda parameter is dynamically adjusted according to the hamming distance to any of a list
         of target sequences such that the closer a sequence gets to any of the targets, the higher its fitness (and the closer lambda gets to 2). Similarly, when the sequence is far away
         from any target, lambda approaches 0.
    Completion is determined by three stopping criteria arguments: %s (see below).
    ''' % ', '.join(['--' + sc for sc in stop_crits])
    parser = argparse.ArgumentParser(description=help_str,
                                     formatter_class=MultiplyInheritedFormatter)

    parser.add_argument('--naive_seq', help='Initial naive nucleotide sequence.')
    parser.add_argument('--naive_seq_file', help='Path to fasta file containing initial naive sequences from which do draw at random.')
    parser.add_argument('--mutability_file', default=file_dir+'/../motifs/Mutability_S5F.csv', help='Path to mutability model file.')
    parser.add_argument('--substitution_file', default=file_dir+'/../motifs/Substitution_S5F.csv', help='Path to substitution model file.')
    parser.add_argument('--no_context', action='store_true', help='Disable context dependence, i.e. use a uniform mutability and substitution.')
    parser.add_argument('--selection', action='store_true', help='If set, simulate with selection (otherwise neutral). Requires that you set --obs_times, and therefore that you *not* set --n_final_seqs.')
    parser.add_argument('--n_final_seqs', type=int, help='If set, simulation stops when we\'ve reached this number of sequences. Note that because sequences with stop codons are subsequently removed, and because more than one sequence is added per iteration, we don\'t necessarily output exactly this many. (If --n_to_sample is also set, then we simulate until we have --n_final_seqs, then downsample to --n_to_sample).')
    parser.add_argument('--obs_times', type=int, nargs='+', help='If set, simulation stops when we\'ve reached this many generations. If more than one value is specified, the largest value is the final observation time (and stopping criterion), and earlier values are used as additional, intermediate sampling times')
    parser.add_argument('--stop_dist', type=int, help='If set, simulation stops when any simulated sequence is closer than this hamming distance from any of the target sequences, according to --metric_for_target_distance.')
    parser.add_argument('--lambda', dest='lambda_', type=float, default=0.9, help='Poisson branching parameter')
    parser.add_argument('--lambda0', type=float, nargs='*', help='Baseline sequence mutation rate(s): first value corresponds to --naive_seq, and optionally the second to --naive_seq2. If only one rate is provided for two sequences, this rate is used for both. If not set, the default is set below')
    parser.add_argument('--target_sequence_lambda0', type=float, default=0.1, help='baseline mutation rate used for generating target sequences (you shouldn\'t need to change this)')
    parser.add_argument('--n_to_sample', type=int, nargs='+', help='Number of cells to sample from the final generation (if one value), or at each generation specified in --obs-times (if same length as --obs_times, and both are set). If --obs_times is set and has more than one value, but --n_to_sample is length one, this same value is applied to each time in --obs_times.')
    parser.add_argument('--kill_sampled_intermediates', action='store_true', help='kill intermediate sequences as they are sampled')
    parser.add_argument('--observe_common_ancestors', action='store_true', help='If set, after deciding which nodes to observe (write to file) according to other options, we then also select the most recent common ancestor for every pair of those nodes (the idea is that this gets you the nodes that you would reconstruct with a phylogenetic program). NOTE histograms written to disk currently don\'t include these.')
    parser.add_argument('--carry_cap', type=int, default=1000, help='The carrying capacity of the simulation with selection. This number affects the fixation time of a new mutation.'
                        'Fixation time is approx. log2(carry_cap), e.g. log2(1000) ~= 10.')
    parser.add_argument('--target_count', type=int, default=10, help='The number of target sequences to generate.')
    parser.add_argument('--target_distance', type=int, default=10, help='Desired distance (using --metric_for_target_distance) between the naive sequence and the target sequences.')
    parser.add_argument('--metric_for_target_distance', default='aa', choices=['aa', 'nuc'], help='Metric to use to calculate the distance to each target sequence (aa: use amino acid distance, i.e. only non-synonymous mutations count, nuc: use nucleotide distance).')
    parser.add_argument('--naive_seq2', help='Second seed naive nucleotide sequence. For simulating heavy/light chain co-evolution.')
    parser.add_argument('--naive_kd', type=float, default=100, help='kd of the naive sequence in nano molar.')
    parser.add_argument('--mature_kd', type=float, default=1, help='kd of the mature sequences in nano molar.')
    parser.add_argument('--skip_update', type=int, default=100, help='Number of leaves/iterations to perform before updating the binding equilibrium (B:A).\n'
                        'The binding equilibrium at any point in time depends, in principle, on the properties of every leaf/cell, and thus would ideally be updated every time any leaf changes.\n'
                        'However, since each individual leaf typically causes only a small change in the overall equilibrium, a substantial speedup with minimal impact on accuracy can be achieved by updating B:A only after modifying every <--skip-update> leaves.\n'
                        '(skip_update < carry_cap/10 recommended.)')
    parser.add_argument('--B_total', type=float, default=1, help='Total number of BCRs per B cell normalized to 10e4. So 1 equals 10e4, 100 equals 10e6 etc. '
                        'It is recommended to keep this as the default.')
    parser.add_argument('--U', type=float, default=5, help='Controls the fraction of BCRs binding antigen necessary to only sustain the life of the B cell '
                        'It is recommended to keep this as the default.')
    parser.add_argument('--f_full', type=float, default=1, help='The fraction of antigen bound BCRs on a B cell that is needed to elicit close to maximum reponse.'
                        'Cannot be smaller than B_total. It is recommended to keep this as the default.')
    parser.add_argument('--k_exp', type=float, default=2, help='The exponent in the function to map hamming distance to kd. '
                        'It is recommended to keep this as the default.')
    parser.add_argument('--plotAA', action='store_true', help='Plot trees with collapsing and coloring on amino acid level.')
    parser.add_argument('--debug', type=int, default=0, choices=[0, 1, 2], help='Debug verbosity level.')
    parser.add_argument('--outbase', default='GCsimulator_out', help='Output file base name')
    parser.add_argument('--idlabel', action='store_true', help='Flag for labeling the sequence ids of the nodes in the output tree images, also write associated fasta alignment if True')
    parser.add_argument('--random_seed', type=int, help='for random number generator')
    parser.add_argument('--pair_bounds', help='for internal use only')
    parser.add_argument('--observe_based_on_affinity', action='store_true', help='When selecting sequences to observe, instead of choosing at random (default), weight with 1/kd (this weighting is kind of arbitrary, and eventually I should maybe figure out something else, but the point is to allow some configurability as to not just sampling entirely at random).')
    parser.add_argument('--verbose', action='store_true', help='DEPRECATED use --debug')
    parser.add_argument('--n_to_downsample', type=int, nargs='+', help='DEPRECATED use --n_to_sample')

    args = parser.parse_args()

    if args.random_seed is not None:
        numpy.random.seed(args.random_seed)
        random.seed(args.random_seed)

    if args.no_context:
        args.mutability_file = None
        args.substitution_file = None
    if args.verbose:
        print('%s transferring deprecated --verbose option to --debug 1' % selection_utils.color('red', 'note:'))
        args.debug = 1
    delattr(args, 'verbose')
    if args.n_to_downsample is not None:
        print('%s transferring deprecated --n_to_downsample option to --n_to_sample' % selection_utils.color('red', 'note:'))
        args.n_to_sample = args.n_to_downsample
    delattr(args, 'n_to_downsample')

    if [args.naive_seq, args.naive_seq_file].count(None) != 1:
        raise Exception('exactly one of --naive_seq and --naive_seq_file must be set')
    if args.naive_seq_file is not None:
        from Bio import SeqIO
        records = list(SeqIO.parse(args.naive_seq_file, "fasta"))
        random.shuffle(records)
        args.naive_seq = str(records[0].seq).upper()
    if args.naive_seq is not None:
        args.naive_seq = args.naive_seq.upper()
    if args.lambda0 is None:
        args.lambda0 = [max([1, int(.01*len(args.naive_seq))])]
    if len(args.naive_seq) % 3 != 0:
        print('  note: padding right side of --naive_seq to multiple of three')
        args.naive_seq += 'N' * (3 - (len(args.naive_seq) % 3))
    args.naive_tseq = TranslatedSeq(args.naive_seq)
    delattr(args, 'naive_seq')  # I think this is the most sensible thing to to
    if has_stop_aa(args.naive_tseq.aa):
        raise Exception('stop codon in --naive_seq (this isn\'t necessarily otherwise forbidden, but it\'ll quickly end you in a thicket of infinite loops, so should be corrected).')

    if [getattr(args, sc) for sc in stop_crits].count(None) == len(stop_crits):
        raise Exception('have to set a stopping criterion (see --help)')
    if args.n_final_seqs is not None and args.n_to_sample is not None:
        if args.n_to_sample[-1] > args.n_final_seqs:
            raise Exception('if both --n_final_seqs and --n_to_sample are set, --n_final_seqs must be larger than the last value in --n_to_sample')
    if args.obs_times is not None and len(args.obs_times) > 1 and args.n_to_sample is None:
        raise Exception('--n_to_sample must be set if more than one obs time is specified, since we don\'t want to just sample all cells at intermediate times')
    if args.n_to_sample is not None and args.obs_times is not None and len(args.n_to_sample) != len(args.obs_times):
        if len(args.n_to_sample) == 1:
            print('  note: expanding --n_to_sample to match --obs_times length: %s --> %s' % (args.n_to_sample, [args.n_to_sample[0] for _ in args.obs_times]))
            args.n_to_sample = [args.n_to_sample[0] for _ in args.obs_times]
        else:
            raise Exception('--n_to_sample has to either be length one (in which case it is automatically expanded to match --obs_times), or has to be the same length as --obs_times')
    if args.obs_times is None and args.n_to_sample is not None and len(args.n_to_sample) > 1:
        raise Exception('doesn\'t make sense to specify multiple values for --n_to_sample if --obs_times is not set (i.e. if we\'re only sampling at the end)')
    if args.obs_times is not None:
        if args.obs_times != sorted(args.obs_times):
            raise Exception('--obs_times must be sorted (we could sort them here, but then you might think you didn\'t need to worry about the order of --n_to_sample being the same)')

    if args.naive_seq2 is not None:
        if len(args.naive_seq2) % 3 != 0:
            print('  note: padding right side of --naive_seq2 to multiple of three')
            args.naive_seq2 += 'N' * (3 - (len(args.naive_seq2) % 3))
        if len(args.lambda0) == 1:  # Use the same mutation rate on both sequences
            args.lambda0 = [args.lambda0[0], args.lambda0[0]]
        elif len(args.lambda0) != 2:
            raise Exception('--lambda0 (set to \'%s\') has to have either two values (if --naive_seq2 is set) or one value (if it isn\'t).' % args.lambda0)
        if len(args.naive_tseq.nuc) % 3 != 0:  # have to pad first one out to a full codon so we don't think there's a bunch of stop codons in the second sequence
            args.naive_tseq.nuc += 'N' * (3 - len(args.naive_tseq.nuc) % 3)
        args.pair_bounds = ((0, len(args.naive_tseq.nuc)), (len(args.naive_tseq.nuc), len(args.naive_tseq.nuc + args.naive_seq2)))  # bounds to allow mashing the two sequences toegether as one string
        args.naive_tseq = TranslatedSeq(args.naive_tseq.nuc + args.naive_seq2.upper())  # merge the two seqeunces to simplify future dealing with the pair:
        if has_stop(args.naive_tseq.nuc):
            raise Exception('stop codon in --naive_seq2 (this isn\'t necessarily otherwise forbidden, but it\'ll quickly end you in a thicket of infinite loops, so should be corrected).')

    if args.selection:
        assert args.target_distance > 0
        assert args.B_total >= args.f_full  # the fully activating fraction on BA must be possible to reach within B_total
        # find the total amount of A necessary for sustaining the specified carrying capacity
        args.A_total = selection_utils.find_A_total(args.carry_cap, args.B_total, args.f_full, args.mature_kd, args.U)
        # calculate the parameters for the logistic function
        args.Lp = selection_utils.find_Lp(args.f_full, args.U)

    run_simulation(args)

# ----------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
