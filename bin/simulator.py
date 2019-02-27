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

from GCutils import hamming_distance, has_stop_aa, translate, CollapsedTree, TranslatedSeq
import selection_utils
from selection_utils import target_distance_fcn

scipy.seterr(all='raise')

# ----------------------------------------------------------------------------------------
# For paired heavy/light, the sequences are stored sequentially in one string. This fcn is for extracting them.
def get_seq_from_pair(joint_seq, pair_bounds, iseq):
    return joint_seq[pair_bounds[iseq][0] : pair_bounds[iseq][1]]

# ----------------------------------------------------------------------------------------
class MutationModel():
    # ----------------------------------------------------------------------------------------
    def __init__(self, args, mutation_order=True, allow_re_mutation=True):
        """
        initialized with input files of the S5F format
        @param mutation_order: whether or not to mutate sequences using a context sensitive manner
                               where mutation order matters
        @param allow_re_mutation: allow the same position to mutate multiple times on a single branch
        """
        self.lambda_min = 10e-10  # small lambdas are causing problems so make a minimum
        self.mutation_order = mutation_order
        self.allow_re_mutation = allow_re_mutation
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
    def init_node(self, args, nuc_seq, time, parent, target_seqs=None):
        node = TreeNode()
        node.add_feature('tseq', TranslatedSeq(nuc_seq))
        node.add_feature('naive_distance', hamming_distance(nuc_seq, args.naive_tseq.nuc))  # always nucleotide distance
        node.add_feature('time', time)
        node.dist = 0 if parent is None else hamming_distance(nuc_seq, parent.tseq.nuc)  # always nucleotide distance (doesn't use add_feature() because it's a builtin ete3 feature)
        node.add_feature('terminated', False)  # set if it's dead (only set if it draws zero children, or if --kill_sampled_intermediates is set and it's sampled at an intermediate time point)
        node.add_feature('intermediate_sampled', False)  # set if it's sampled at an intermediate time point
        node.add_feature('frequency', 0)  # observation frequency, is set to either 1 or 0 in set_observation_frequencies_and_names(), then when the collapsed tree is constructed it can get bigger than 1 when the frequencies of nodes connected by zero-length branches are summed
        if args.selection:
            node.add_feature('lambda_', None)  # set in selection_utils.update_lambda_values() (i.e. is modified every (few) generations)
            node.add_feature('target_distance', target_distance_fcn(args, node.tseq, target_seqs))  # nuc or aa distance, depending on args
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
    def mutate(self, sequence, lambda0, return_n_mutations=False, debug=False):
        """
        Mutate a sequence, with lamdba0 the baseline mutability.
        Cannot mutate the same position multiple times.
        @param sequence: the original sequence to mutate
        @param lambda0: a "baseline" mutation rate
        """

        mutabilities = None
        sequence_mutability = 1.
        if self.context_model is not None:
            mutabilities = self.mutabilities(sequence)
            sequence_mutability = sum(mty[0] for mty in mutabilities) / len(sequence)
        lambda_sequence = sequence_mutability * lambda0  # Poisson rate for this sequence (given its relative mutability):

        # decide how many mutations we'll apply
        n_mutations = numpy.random.poisson(lambda_sequence)
        if not self.allow_re_mutation:
            trials = 20
            for trial in range(1, trials+1):
                n_mutations = numpy.random.poisson(lambda_sequence)
                if n_mutations <= len(sequence):
                    break
                if trial == trials:
                    raise RuntimeError('mutations saturating, consider reducing lambda0')

        # Introduce mutations (note: we very commonly just return, i.e. if the poisson kicks up zero mutations)
        unmutated_positions = range(len(sequence))
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
            nucs = [n for n in 'ACGT' if n != sequence[mut_pos]]
            substitution_p = None
            if self.context_model is not None:
                substitution_p = [mutabilities[mut_pos][1][n] for n in nucs]
                assert 0 <= abs(sum(substitution_p) - 1.) < 1e-10
            new_nuc = scipy.random.choice(nucs, p=substitution_p)
            if debug:
                dbg_str += ['%s%d%s' % (sequence[mut_pos], mut_pos, new_nuc)]
            sequence = sequence[ : mut_pos] + new_nuc + sequence[mut_pos + 1 :]
            if self.context_model is not None and self.mutation_order:  # If mutation order matters, the mutabilities of the sequence need to be updated
                mutabilities = self.mutabilities(sequence)
            if not self.allow_re_mutation:  # Remove this position so we don't mutate it again
                unmutated_positions.remove(mut_pos)

        if debug:
            print('  '.join(dbg_str))
        if return_n_mutations:
            return sequence, n_mutations
        else:
            return sequence

    # ----------------------------------------------------------------------------------------
    # Make a single target sequence with <n_muts> hamming distance from <args.naive_tseq> (nuc or aa distance according to args.metric_for_target_distance)
    def make_target_sequence(self, args, n_max_tries=100):
        assert not has_stop_aa(args.naive_tseq.aa)  # already checked during argument parsing, but we really want to make sure since the loop will blow up if there's a stop to start with
        itry = 0
        while itry < n_max_tries:
            dist = None
            tseq = args.naive_tseq
            while dist is None or dist < args.target_distance:
                tseq = TranslatedSeq(self.mutate(tseq.nuc, args.target_sequence_lambda0))
                dist = target_distance_fcn(args, args.naive_tseq, [tseq])
                if dist == args.target_distance and not has_stop_aa(tseq.aa):  # Stop codon cannot be part of the return
                    return tseq
            itry += 1

        raise RuntimeError('fell through after trying %d times to make a target sequence' % n_max_tries)

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
    def get_hdist_hist(self, args, leaves):
        bin_edges = list(numpy.arange(-0.5, int(2 * args.target_distance) + 0.5))  # some sequences manage to wander quite far away from their nearest target without getting killed, so multiply by 2
        hist = scipy.histogram([l.target_distance for l in leaves], bins=bin_edges)  # if <bins> is a list, it defines the bin edges, including the rightmost edge
        return hist

    # ----------------------------------------------------------------------------------------
    def simulate(self, args):
        '''
        Simulate a poisson branching process with mutation introduced
        by the chosen mutation model e.g. motif or uniform.
        Can either simulate under a neutral model without selection,
        or using an affinity muturation inspired model for selection.
        '''

        if args.selection:
            self.sampled_hdist_hists, self.hdist_hists = [None for _ in range(max(args.obs_times) + 1)], [None for _ in range(max(args.obs_times) + 1)]  # list (over generations) of histograms of min AA distance to target over leaves
            self.n_mutated_hists = [None for _ in range(max(args.obs_times) + 1)]

            print('    making %d target sequences' % args.target_count)
            target_seqs = [self.make_target_sequence(args) for i in range(args.target_count)]
            with open(args.outbase + '_targets.fa', 'w') as tfile:
                for itarget, tseq in enumerate(target_seqs):
                    tfile.write('>%s\n%s\n' % ('target-%d' % itarget, tseq.nuc))

            assert len(set([len(args.naive_tseq.aa)] + [len(t.aa) for t in target_seqs])) == 1  # targets and naive seq are same length
            assert len(set([args.target_distance] + [target_distance_fcn(args, args.naive_tseq, [t]) for t in target_seqs])) == 1  # all targets are the right distance from the naive

        tree = self.init_node(args, args.naive_tseq.nuc, 0, None, target_seqs)
        if args.selection:
            self.hdist_hists[0] = self.get_hdist_hist(args, tree)

        print('    starting %d generations' % max(args.obs_times))
        if args.verbose:
            print('       gen   live leaves')
            print('             before/after   ileaf   n children     n mutations          Kd   (%s: updated lambdas)' % selection_utils.color('blue', 'x'))
        current_time = 0
        n_unterminated_leaves = 1
        while True:
            if n_unterminated_leaves <= 0:  # if everybody's dead (probably can't actually go less than zero, but not sure)
                break
            if args.n_final_seqs is not None and n_unterminated_leaves >= args.n_final_seqs:  # if we've got as many sequences as we were asked for
                break
            if args.obs_times is not None and current_time >= max(args.obs_times):  # if we've done as many generations as we were told to
                break
            if args.stop_dist is not None and current_time > 0 and min([l.target_distance for l in tree.iter_leaves()]) <= args.stop_dist:  # if the leaves have gotten close enough to the target sequences
                break

            # sample any requested intermediate time points (from *last* generation, since we haven't yet incremented current_time)
            if args.obs_times is not None and len(args.obs_times) > 1 and current_time in args.obs_times:
                assert len(args.obs_times) == len(args.n_to_downsample)
                n_to_sample = args.n_to_downsample[args.obs_times.index(current_time)]
                live_nostop_leaves = [l for l in tree.iter_leaves() if not l.terminated and not has_stop_aa(l.tseq.aa)]
                if len(live_nostop_leaves) < n_to_sample:
                    raise RuntimeError('tried to sample %d leaves at intermediate timepoint %d, but tree only has %d live leaves without stops (try sampling at a later generation, or use a larger carrying capacity).' % (n_to_sample, current_time, len(live_nostop_leaves)))
                inter_sampled_leaves = self.choose_leaves_to_sample(args, live_nostop_leaves, n_to_sample)
                for leaf in inter_sampled_leaves:
                    leaf.intermediate_sampled = True
                    if args.kill_sampled_intermediates:
                        leaf.terminated = True
                        n_unterminated_leaves -= 1
                self.sampled_hdist_hists[current_time] = self.get_hdist_hist(args, inter_sampled_leaves)
                print('                  sampled %d (of %d live and stop-free) intermediate leaves (%s) at time %d (but time is about to increment to %d)' % (n_to_sample, len(live_nostop_leaves), 'killing each of them' if args.kill_sampled_intermediates else 'leaving them alive', current_time, current_time + 1))

            current_time += 1

            skip_lambda_n = 0  # index keeping track of how many leaves since we last updated all the lambdas
            live_leaves = [l for l in tree.iter_leaves() if not l.terminated]  # NOTE this is out of date as soon as we've added any children in the loop, or killed anybody with no children
            updated_live_leaves = [l for l in live_leaves]  # but this one, we keep updating (so we don't have to call iter_leaves() so much, which was taking quite a bit of time) (matches n_unterminated_leaves)
            random.shuffle(live_leaves)
            if args.verbose:
                print('      %3d    %3d' % (current_time, len(live_leaves)), end='\n' if len(live_leaves) == 0 else '')  # NOTE these are live leaves *after* the intermediate sampling above
            for leaf in live_leaves:
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
                if current_time == 1 and len(live_leaves) == 1:  # if the first leaf draws zero children, keep trying so we don't have to throw out the whole tree and start over
                    itry = 0
                    while n_children == 0:
                        n_children = numpy.random.poisson(this_lambda)
                        itry += 1
                        if itry > 10 == 0:
                            print('too many tries to get at least one child, giving up on tree')
                            break

                n_unterminated_leaves += n_children - 1
                if n_children == 0:  # kill everyone with no children
                    leaf.terminated = True
                    updated_live_leaves.remove(leaf)
                    if len(live_leaves) == 1:
                        print('  terminating only leaf with no children')

                if args.verbose:
                    n_mutation_list, kd_list = [], []
                for _ in range(n_children):
                    if args.naive_seq2 is not None:  # for paired heavy/light we mutate them separately with their own mutation rate
                        mutated_sequence1 = self.mutate(get_seq_from_pair(leaf.tseq.nuc, args.pair_bounds, iseq=0), args.lambda0[0])
                        mutated_sequence2 = self.mutate(get_seq_from_pair(leaf.tseq.nuc, args.pair_bounds, iseq=1), args.lambda0[1])
                        mutated_sequence = mutated_sequence1 + mutated_sequence2
                    else:
                        mutated_sequence, n_muts = self.mutate(leaf.tseq.nuc, args.lambda0[0], return_n_mutations=True)
                        if args.verbose:
                            n_mutation_list.append(n_muts)
                    child = self.init_node(args, mutated_sequence, current_time, leaf, target_seqs)
                    if args.selection and args.verbose:
                        kd_list.append(child.Kd)
                    leaf.add_child(child)
                    updated_live_leaves.append(child)
                    if leaf in updated_live_leaves:  # <leaf> isn't a leaf any more, since now it has children
                        updated_live_leaves.remove(leaf)  # now that it's been updated, it matches n_unterminated_leaves
                if args.verbose:
                    n_mutation_str_list = [('%d' % n) if n > 0 else '-' for n in n_mutation_list]
                    kd_str_list = ['%.0f' % kd for kd in kd_list]
                    pre_leaf_str = '' if live_leaves.index(leaf) == 0 else ('%12s %3d' % ('', len(updated_live_leaves)))
                    print(('      %s      %4d   %3d  %s          %-14s       %-28s') % (pre_leaf_str, live_leaves.index(leaf), n_children, lambda_update_dbg_str, ' '.join(n_mutation_str_list), ' '.join(kd_str_list)))
            if args.selection:
                self.hdist_hists[current_time] = self.get_hdist_hist(args, updated_live_leaves)
                n_mutated_hdists = [l.naive_distance for l in updated_live_leaves]  # nuc distance to naive sequence
                self.n_mutated_hists[current_time] = scipy.histogram(n_mutated_hdists, bins=list(numpy.arange(-0.5, max(args.obs_times) + 0.5)))  # can't have more than one mutation per generation

        # write a histogram of the hamming distances to target at each generation
        if args.selection:
            with open(args.outbase + '_min_aa_target_hdists.p', 'wb') as histfile:
                pickle.dump(self.hdist_hists, histfile)
            with open(args.outbase + '_n_mutated_nuc_hdists.p', 'wb') as histfile:
                pickle.dump(self.n_mutated_hists, histfile)

        # check some things
        if args.obs_times is not None and max(args.obs_times) != current_time:
            raise RuntimeError('tree terminated at time %d, but we were supposed to sample at time %d' % (current_time, max(args.obs_times)))
        if args.n_final_seqs is not None and n_unterminated_leaves < args.n_final_seqs:
            raise RuntimeError('tree terminated with %d leaves, but --n_final_seqs was set to %d' % (n_unterminated_leaves, args.n_final_seqs))
        if args.n_to_downsample is not None and n_unterminated_leaves < args.n_to_downsample[-1]:
            raise RuntimeError('tree terminated with %d leaves, but --n_to_downsample[-1] was set to %d' % (n_unterminated_leaves, args.n_to_downsample[-1]))
        if args.obs_times is not None and len(args.obs_times) > 1:  # make sure we have the right number of sampled intermediates at each intermediate time point
            for inter_time, n_to_sample in zip(args.obs_times[:-1], args.n_to_downsample[:-1]):
                intermediate_sampled_leaves = [l for l in tree.iter_descendants() if l.time == inter_time and l.intermediate_sampled]  # nodes at this time point that we sampled above
                if len(intermediate_sampled_leaves) < n_to_sample:
                    raise RuntimeError('couldn\'t find the correct number of intermediate sampled leaves at time %d (should have sampled %d, but now we only find %d)' % (inter_time, n_to_sample, len(intermediate_sampled_leaves)))

        stop_leaves = [l for l in tree.iter_leaves() if l.time == current_time and has_stop_aa(l.tseq.aa)]
        non_stop_leaves = [l for l in tree.iter_leaves() if l.time == current_time and not has_stop_aa(l.tseq.aa)]  # non-stop leaves
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
        if args.n_to_downsample is not None and len(observed_leaves) > args.n_to_downsample[-1]:  # if we were asked to downsample, and if there's enough leaves to do so
            observed_leaves = self.choose_leaves_to_sample(args, observed_leaves, args.n_to_downsample[-1])

        for leaf in observed_leaves:
            leaf.frequency = 1
            uid, potential_names, used_names = selection_utils.choose_new_uid(potential_names, used_names)
            leaf.name = 'leaf-' + uid

        self.sampled_hdist_hists[current_time] = self.get_hdist_hist(args, observed_leaves)  # NOTE this doesn't nodes added from --observe_common_ancestors or --observe_all_ancestors

        if len(self.sampled_hdist_hists) > 0:
            with open(args.outbase + '_sampled_min_aa_target_hdists.p', 'wb') as histfile:
                pickle.dump(self.sampled_hdist_hists, histfile)

        # prune away lineages that have zero total observation frequency
        for node in tree.iter_descendants():
            if sum(child.frequency for child in node.traverse()) == 0:  # if all children of <node> have zero observation frequency, detach <node> (only difference between traverse() and iter_descendants() seems to be that traverse() includes the node on which you're calling it, while iter_descendants() doesn't)
                node.detach()

        # remove unobserved unifurcations
        for node in tree.iter_descendants():
            parent = node.up
            if node.frequency == 0 and len(node.children) == 1:
                node.delete(prevent_nondicotomic=False)  # seems like this should instead use the preserve_branch_length=True option so we don't need the next line, but I don't want to change it
                node.children[0].dist = hamming_distance(node.children[0].tseq.nuc, parent.tseq.nuc)

        if args.observe_common_ancestors:
            n_observed_ancestors = 0
            for ancestor in tree.traverse():
                if ancestor.is_leaf():
                    continue
                ancestor.frequency = 1
                uid, potential_names, used_names = selection_utils.choose_new_uid(potential_names, used_names)
                ancestor.name = 'mrca-' + uid
                n_observed_ancestors += 1
            print('    added %d ancestor nodes' % n_observed_ancestors)

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
            fh.write('>%s\n%s\n' % (tree.name, get_seq_from_pair(args.naive_tseq.nuc, args.pair_bounds, iseq=iseq)))
        for node in [n for n in tree.iter_descendants() if n.frequency != 0]:  # NOTE doesn't iterate over root node
            for iseq, fh in enumerate(fhandles):
                fh.write('>%s\n%s\n' % (node.name, get_seq_from_pair(node.tseq.nuc, args.pair_bounds, iseq=iseq)))
    else:
        with open('%s.fasta' % args.outbase, 'w') as fh:
            fh.write('>%s\n%s\n' % (tree.name, args.naive_tseq.nuc))
            for node in [n for n in tree.iter_descendants() if n.frequency != 0]:  # NOTE doesn't iterate over root node
                fh.write('>%s\n%s\n' % (node.name, node.tseq.nuc))

    # write some observable simulation stats
    frequency, distance_from_naive, degree = zip(*[(node.frequency,
                                                    node.naive_distance,
                                                    sum(hamming_distance(node.tseq.nuc, node2.tseq.nuc) == 1 for node2 in collapsed_tree.tree.traverse() if node2.frequency and node2 is not node))
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
        colors[tree.tseq.aa] = 'gray'
    else:
        colors[tree.tseq.nuc] = 'gray'

    for n in tree.traverse():
        nstyle = NodeStyle()
        nstyle["size"] = 10
        if args.plotAA:
            if n.tseq.aa not in colors:
                colors[n.tseq.aa] = next(palette)
            nstyle['fgcolor'] = colors[n.tseq.aa]
        else:
            if n.tseq.nuc not in colors:
                colors[n.tseq.nuc] = next(palette)
            nstyle['fgcolor'] = colors[n.tseq.nuc]
        n.set_style(nstyle)

    # Render and pickle lineage tree:
    tree.render(args.outbase+'_lineage_tree.svg', tree_style=ts)
    with open(args.outbase+'_lineage_tree.p', 'wb') as f:
        pickle.dump(tree, f)

    # Render collapsed tree,
    # create an id-wise colormap
    # NOTE: node.name can be a set
    if args.plotAA and args.selection:
        colormap = {node.name:colors[node.tseq.aa] for node in collapsed_tree.tree.traverse()}
    else:
        colormap = {node.name:colors[node.tseq.nuc] for node in collapsed_tree.tree.traverse()}
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
            hdist_hists = pickle.load(fh)
            selection_utils.plot_runstats(hdist_hists, args.outbase, colors)


# ----------------------------------------------------------------------------------------
def main():
    import argparse
    file_path = os.path.realpath(__file__)
    file_dir = os.path.dirname(file_path)

    class MultiplyInheritedFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass
    formatter_class = MultiplyInheritedFormatter
    help_str = '''
    Germinal center simulation. Can simulate in two modes:
      a) Neutral mode. A Galton–Watson process, with mutation probabilities according to a user defined motif model e.g. S5F.
      b) Selection mode. With the same mutation process as in a), but the poisson progeny distribution's lambda parameter is dynamically adjusted according to the hamming distance to any of a list
         of target sequences such that the closer a sequence gets to any of the targets, the higher its fitness (and the closer lambda gets to 2). Similarly, when the sequence is far away
         from any target, lambda approaches 0.
    Completion is determined by the three arguments --n_final_seqs, --obs_times, and --stop_dist (see below).
    '''
    parser = argparse.ArgumentParser(description=help_str,
                                     formatter_class=MultiplyInheritedFormatter)

    parser.add_argument('--naive_seq', help='Initial naive nucleotide sequence.')
    parser.add_argument('--naive_seq_file', help='Path to fasta file containing initial naive sequences from which do draw at random.')
    parser.add_argument('--mutability_file', default=file_dir+'/../motifs/Mutability_S5F.csv', help='Path to mutability model file.')
    parser.add_argument('--substitution_file', default=file_dir+'/../motifs/Substitution_S5F.csv', help='Path to substitution model file.')
    parser.add_argument('--no_context', action='store_true', help='Disable context dependence, i.e. use a uniform mutability and substitution.')
    parser.add_argument('--selection', action='store_true', help='If set, simulate with selection (otherwise neutral). Requires that you set --obs_times, and therefore that you *not* set --n_final_seqs.')
    parser.add_argument('--n_final_seqs', type=int, help='If set, simulation stops when we\'ve reached this number of sequences (other stopping criteria: --stop_dist and --obs_times). Because sequences with stop codons are subsequently removed, and because more than on sequence is added per iteration, though we don\'t necessarily output this many. (If --n_to_downsample is also set, then we simulate until we have --n_final_seqs, then downsample to --n_to_downsample).')
    parser.add_argument('--obs_times', type=int, nargs='+', help='If set, simulation stops when we\'ve reached this many generations. If more than one value is specified, the largest value is the final observation time (and stopping criterion), and earlier values are used as additional, intermediate sampling times (other stopping criteria: --n_final_seqs, --stop_dist)')
    parser.add_argument('--stop_dist', type=int, help='If set, simulation stops when any simulated sequence is closer than this hamming distance from any of the target sequences, according to --metric_for_target_distance (other stopping criteria: --n_final_seqs, --obs_times).')
    parser.add_argument('--lambda', dest='lambda_', type=float, default=0.9, help='Poisson branching parameter')
    parser.add_argument('--lambda0', type=float, nargs='*', help='Baseline sequence mutation rate(s): first value corresponds to --naive_seq, and optionally the second to --naive_seq2. If only one rate is provided for two sequences, this rate is used for both. If not set, the default is set below')
    parser.add_argument('--target_sequence_lambda0', type=float, default=0.1, help='baseline mutation rate used for generating target sequences (you shouldn\'t need to change this)')
    parser.add_argument('--n_to_downsample', type=int, nargs='+', help='Number of cells sampled during each sampling step. If one value is specified, this same value is applied to each time in --obs_times; whereas if more than one value is specified, each is applied to the corresponding value in --obs_times.')
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
    parser.add_argument('--verbose', action='store_true', help='Print progress during simulation. Mostly useful for simulation with selection since this can take a while.')
    parser.add_argument('--outbase', default='GCsimulator_out', help='Output file base name')
    parser.add_argument('--idlabel', action='store_true', help='Flag for labeling the sequence ids of the nodes in the output tree images, also write associated fasta alignment if True')
    parser.add_argument('--random_seed', type=int, help='for random number generator')
    parser.add_argument('--pair_bounds', help='for internal use only')
    parser.add_argument('--observe_based_on_affinity', action='store_true', help='When selecting sequences to observe, instead of choosing at random (default), weight with 1/kd (this weighting is kind of arbitrary, and eventually I should maybe figure out something else, but the point is to allow some configurability as to not just sampling entirely at random).')

    args = parser.parse_args()
    if args.random_seed is not None:
        numpy.random.seed(args.random_seed)
        random.seed(args.random_seed)
    if args.no_context:
        args.mutability_file = None
        args.substitution_file = None
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
    args.naive_tseq = TranslatedSeq(args.naive_seq)
    args.naive_seq = None  # I think this is the most sensible thing to to
    if has_stop_aa(args.naive_tseq.aa):
        raise Exception('stop codon in --naive_seq (this isn\'t necessarily otherwise forbidden, but it\'ll quickly end you in a thicket of infinite loops, so should be corrected).')

    if [args.n_final_seqs, args.obs_times].count(None) != 1:
        raise Exception('exactly one of --n_final_seqs and --obs_times must be set')
    if args.selection and args.obs_times is None:
        raise Exception('--obs_times must be specified if --selection is set')
    if args.n_final_seqs is not None and args.n_to_downsample is not None and args.n_to_downsample[-1] > args.n_final_seqs:
        raise Exception('if both --n_final_seqs and --n_to_downsample are set, --n_to_downsample must be larger (than the last value in --n_to_downsample)')
    if args.n_final_seqs is not None and args.n_to_downsample is not None and len(args.n_to_downsample) != 1:
        raise Exception('--n_to_downsample must a single value when specifying --n_final_seqs')
    if args.obs_times is not None and len(args.obs_times) > 1 and args.n_to_downsample is None:
        raise Exception('--n_to_downsample must be specified when using intermediate sampling')
    if args.n_to_downsample is not None and args.obs_times is not None and len(args.n_to_downsample) != len(args.obs_times):
        if len(args.n_to_downsample) == 1:
            print('  note: expanding --n_to_downsample to match --obs_times length: %s --> %s' % (args.n_to_downsample, [args.n_to_downsample[0] for _ in args.obs_times]))
            args.n_to_downsample = [args.n_to_downsample[0] for _ in args.obs_times]
        else:
            raise Exception('--n_to_downsample has to either be length one (in which case it is automatically to match --obs_times), or has to be the same length as --obs_times')
    if args.obs_times is not None:
        if args.obs_times != sorted(args.obs_times):
            raise Exception('--obs_times must be sorted (we could sort them here, but then you might think you didn\'t need to worry about the order of --n_to_downsample being the same)')
            # args.obs_times = sorted(args.obs_times)

    if args.naive_seq2 is not None:
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
