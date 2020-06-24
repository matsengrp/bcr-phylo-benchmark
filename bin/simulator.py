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

from GCutils import hamming_distance, has_stop_aa, local_translate, replace_codon_in_aa_seq, CollapsedTree, TranslatedSeq
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
            self.translation_cache[nuc_seq] = local_translate(nuc_seq)
        return self.translation_cache[nuc_seq]

    # ----------------------------------------------------------------------------------------
    def init_node(self, args, nuc_seq, time, parent, target_seqs=None, aa_seq=None, mean_kd=None):
        node = TreeNode()
        node.add_feature('nuc_seq', nuc_seq)  # NOTE don't use a TranslatedSeq feature since it gets written to pickle files, which then requires importing the class definition
        node.add_feature('aa_seq', aa_seq if aa_seq is not None else self.get_translation(nuc_seq))
        node.add_feature('naive_distance', hamming_distance(nuc_seq, args.naive_tseq.nuc))  # always nucleotide distance
        node.add_feature('time', time)
        node.dist = 0 if parent is None else hamming_distance(nuc_seq, parent.nuc_seq)  # always nucleotide distance (doesn't use add_feature() because it's a builtin ete3 feature)
        node.add_feature('terminated', False)  # set if it's dead (only set if it draws zero children, or if --kill_sampled_intermediates is set and it's sampled at an intermediate time point)
        node.add_feature('frequency', 0)  # observation frequency, is set to either 1 or 0 in set_observation_frequencies_and_names(), then when the collapsed tree is constructed it can get bigger than 1 when the frequencies of nodes connected by zero-length branches are summed
        if args.selection:
            node.add_feature('lambda_', None)  # set in selection_utils.update_lambda_values() (i.e. is modified every few generations)
            itarget, tdist = target_distance_fcn(args, TranslatedSeq(args, nuc_seq, aa_seq=node.aa_seq), target_seqs)
            node.add_feature('target_index', itarget)
            node.add_feature('target_distance', tdist)  # nuc or aa distance, depending on args
            node.add_feature('Kd', selection_utils.calc_kd(node, args))
            node.add_feature('relative_Kd', node.Kd / float(mean_kd) if mean_kd is not None else None)  # kd relative to mean of the current leaves
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
    def make_target_sequence(self, args, initial_tseq, tdist, lambda0, n_max_tries=100):
        assert not has_stop_aa(initial_tseq.aa)  # already checked during argument parsing, but we really want to make sure since the loop will blow up if there's a stop to start with
        itry = 0
        while itry < n_max_tries:
            dist = None
            tseq = initial_tseq
            while dist is None or dist < tdist:
                mfo = self.mutate(tseq.nuc, lambda0, aa_seq=tseq.aa)
                tseq = TranslatedSeq(args, mfo['nuc_seq'], aa_seq=mfo['aa_seq'])
                _, dist = target_distance_fcn(args, initial_tseq, [tseq])
                # TODO wouldn't it make more sense (or at least be faster) to give up as soon as you get a stop codon?
                if dist >= tdist and not has_stop_aa(tseq.aa):  # greater-than is for aa-sim metric, since it's almost continuous
                    return tseq
            itry += 1

        raise Exception('couldn\'t make a target sequence in %d tries (didn\'t make it to requested target distance %d, and/or had stops)' % (n_max_tries, tdist))

    # ----------------------------------------------------------------------------------------
    def get_targets(self, args):
        start = time.time()

        main_target_count = args.target_count if args.n_target_clusters is None else args.n_target_clusters
        if args.n_target_clusters is not None:
            print('      making %d main target sequences' % main_target_count)
        target_seqs = [self.make_target_sequence(args, args.naive_tseq, args.target_distance, args.target_sequence_lambda0) for i in range(main_target_count)]
        assert len(set([len(args.naive_tseq.aa)] + [len(t.aa) for t in target_seqs])) == 1  # targets and naive seq are same length

        if args.n_target_clusters is not None:
            tmp_n_per_cluster = max(1, int(args.target_count / float(args.n_target_clusters)))  # you should really set them so they are nicely divisible integers, but if you don't, you'll still get at least one, and some kind of rounding (the goal here is to have --target_count always be the total number of target sequences)
            cluster_sizes = [tmp_n_per_cluster for _ in target_seqs]
            if sum(cluster_sizes) > args.target_count:
                raise Exception('inconsistent --n_target_clusters %d and --target_count %d' % (args.n_target_clusters, args.target_count))
            if sum(cluster_sizes) < args.target_count:
                cluster_sizes[-1] += args.target_count - sum(cluster_sizes)
            assert sum(cluster_sizes) == args.target_count
            final_target_seqs = []
            for tseq, csize in zip(target_seqs, cluster_sizes):
                cluster_targets = [self.make_target_sequence(args, tseq, args.target_cluster_distance, args.target_sequence_lambda0) for i in range(csize - 1)]
                final_target_seqs += [tseq] + cluster_targets
            target_seqs = final_target_seqs
            print('      added clusters around each main target for %d total targets: %s' % (len(target_seqs), ' '.join(str(cs) for cs in cluster_sizes)))

        with open(args.outbase + '_targets.fa', 'w') as tfile:
            for itarget, tseq in enumerate(target_seqs):
                tfile.write('>%s\n%s\n' % ('target-%d' % itarget, tseq.nuc))  # [:len(tseq.nuc) - args.n_pads_added]))

        tdists = [target_distance_fcn(args, args.naive_tseq, [t])[1] for t in target_seqs]
        print('    made %d total target seqs in %.1fs with distances %s  (asked for %.1f)' % (len(target_seqs), time.time()-start, ' '.join(['%.1f' % d for d in tdists]), args.target_distance))  # oh, wait, maybe this doesn't take any real time any more? i thought it used to involve more iteration/traversing
        if len(set([args.target_distance] + tdists)) > 1 and 'aa-sim' not in args.metric_for_target_distance:  # can't require it to be exactly equal for aa-sim, since the distance between various amino acids varies close to continuously
            print('    %s target distances not all equal to requested distance' % selection_utils.color('red', 'note'))
        return target_seqs

    # ----------------------------------------------------------------------------------------
    def choose_leaves_to_sample(self, args, leaves, n_to_sample):
        if args.leaf_sampling_scheme == 'uniform-random':
            return random.sample(leaves, n_to_sample)
        elif args.leaf_sampling_scheme == 'affinity-biased':
            probs = [1. / l.Kd for l in leaves]
            total = sum(probs)
            probs = [p / total for p in probs]
            return list(numpy.random.choice(leaves, n_to_sample, p=probs, replace=False))
        elif args.leaf_sampling_scheme == 'high-affinity':
            sorted_leaves = sorted(leaves, key=lambda l: l.Kd)
            return sorted_leaves[:n_to_sample]
        else:
            raise Exception('unsupported leaf sampling scheme %s' % args.leaf_sampling_scheme)

    # ----------------------------------------------------------------------------------------
    def get_target_distance_hist(self, args, leaves):
        if not args.selection:
            return scipy.histogram([0])
        bin_edges = list(numpy.arange(-0.5, int(2 * args.target_distance) + 0.5))  # some sequences manage to wander quite far away from their nearest target without getting killed, so multiply by 2
        hist = scipy.histogram([l.target_distance for l in leaves], bins=bin_edges)  # if <bins> is a list, it defines the bin edges, including the rightmost edge
        return hist

    # ----------------------------------------------------------------------------------------
    def get_ancestor_above_leaf_to_detach(self, dead_leaf):  # this is kind of a lot of infrastructure, but it avoids the old way of finding all these lineages afterward, when the tree is huuuuge, which gets really slow as the tree gets large (e.g. for larger times)
        # note, also, that we could do the actual pruning/detaching here, but for now I think it's better to have the entire tree available at the end of the main loop, in case we want to do something with it, rather than removing bits of it as we go
        parent, last_parent = dead_leaf, None
        while parent.time != 0:  # propagate the termination information up the tree, so we don't have to do so much descendent iteration after the loop when we remove dead lineages
            if parent in self.intermediate_sampled_lineage_nodes:
                break
            if any(not s.terminated for s in parent.children):  # stop if any siblings are unterminated (will be the first branching, except if we've already killed off the sibling lineage in a previous generation)
                break
            parent.terminated = True  # this is necessary to avoid cases where we follow two parallel lineages up, and both return the same <last_parent> (since the second one we do doesn't know that we already did the first one unless we set terminated here)
            last_parent = parent
            parent = parent.up
        if last_parent is not None:
            self.nodes_to_detach.add(last_parent)  # wait to actually detach them til we've finished some other stuff

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
            self.intermediate_sampled_nodes.append(leaf)
            parent = leaf
            while not parent.is_root():
                self.intermediate_sampled_lineage_nodes.add(parent)
                parent = parent.up
            if args.kill_sampled_intermediates:
                leaf.terminated = True
                self.n_unterminated_leaves -= 1
                self.get_ancestor_above_leaf_to_detach(leaf)
        self.sampled_tdist_hists[current_time] = self.get_target_distance_hist(args, inter_sampled_leaves)
        print('                  sampled %d (of %d live and stop-free) intermediate leaves (%s) at end of generation %d (sampling scheme: %s)' % (n_to_sample, len(live_nostop_leaves), 'killing each of them' if args.kill_sampled_intermediates else 'leaving them alive', current_time, args.leaf_sampling_scheme))

    # ----------------------------------------------------------------------------------------
    def check_termination(self, args, current_time, updated_live_leaves):
        finished, successful = False, False
        dbgstr, termstr = [], []
        if self.n_unterminated_leaves <= 0:  # if everybody's dead (probably can't actually go less than zero, but not sure)
            termstr += ['    stopping: no unterminated leaves']
            finished = True
            successful = False
        if len([l.Kd for l in updated_live_leaves if l.Kd != float('inf')]) == 0:  # if everybody's got a stop codon (at least a.t.m. that's the only way you get inf kd)
            termstr += ['    stopping: all leaves have infinite kd']
            finished = True
            successful = False
        if args.n_final_seqs is not None:
            dbgstr += ['n leaves %d' % self.n_unterminated_leaves]
            if self.n_unterminated_leaves >= args.n_final_seqs:  # if we've got as many sequences as we were asked for
                termstr += ['    --n_final_seqs: breaking with %d >= %d unterminated leaves' % (self.n_unterminated_leaves, args.n_final_seqs)]
                finished = True
                successful = True
        if args.obs_times is not None:
            dbgstr += ['time %d' % current_time]
            if current_time >= max(args.obs_times):  # if we've done as many generations as we were told to
                termstr += ['    --obs_times: breaking at generation %d >= %d' % (current_time, max(args.obs_times))]
                finished = True
                successful = True
        if args.stop_dist is not None and current_time > 0:
            min_tdist = min([l.target_distance for l in updated_live_leaves])
            dbgstr += ['min tdist %d' % min_tdist]
            if min_tdist <= args.stop_dist:  # if the leaves have gotten close enough to the target sequences
                termstr += ['    --stop_dist: breaking with min target distance %d <= %d' % (min_tdist, args.stop_dist)]
                finished = True
                successful = True
        if finished and self.n_unterminated_leaves < 2:  # moving this check from later on, so we can rerun if we finish successfully but only have one leaf (I think some of the post-processing steps must fail if there's only one leaf)
            successful = False
        return finished, successful, ', '.join(dbgstr), '\n'.join(termstr)

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
        self.intermediate_sampled_nodes = []  # actual intermediate-sampled nodes
        self.intermediate_sampled_lineage_nodes = set()  # any nodes ancestral to intermediate-sampled nodes (we keep track of these so we know not to prune them)
        self.nodes_to_detach = set()
        tree = self.init_node(args, args.naive_tseq.nuc, 0, None, target_seqs, mean_kd=args.naive_kd)
        if args.selection:
            self.tdist_hists[0] = self.get_target_distance_hist(args, tree)  # i guess the first entry in the other two just stays None

        if args.debug == 1:
            print('        end of      live     target dist (%s)          kd            lambda       termination' % args.metric_for_target_distance)
            print('      generation   leaves      min  mean           min    mean      max  mean        checks')
        if args.debug > 1:
            print('       gen   live leaves   (%s: terminated/no children)' % selection_utils.color('red', 'x'))
            print('             before/after   ileaf   lambda  n children     n mutations            Kd   (%s: updated lambdas)' % selection_utils.color('blue', 'x'))
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
            updated_mean_kd = None  # keep track of mean kd over current leaves, so on initialization we can set each leaf's relative_kd
            updated_lambda_values = None  # just for dbg (need to keep track of it since we update them only periodically)
            random.shuffle(static_live_leaves)
            if args.debug > 1:
                print('      %3d    %3d' % (current_time, len(static_live_leaves)), end='\n' if len(static_live_leaves) == 0 else '')  # NOTE these are live leaves *after* the intermediate sampling above
            for leaf in static_live_leaves:
                if args.selection:
                    lambda_update_dbg_str = ' '
                    if skip_lambda_n == 0:  # time to update the lambdas for every leaf
                        skip_lambda_n = args.skip_update + 1  # reset it so we don't update again until we've done <args.skip_update> more leaves (+ 1 is so that if args.skip_update is 0 we don't skip at all, i.e. args.skip_update is the number of leaves skipped, *not* the number of leaves *between* updates)
                        updated_lambda_values = selection_utils.update_lambda_values(updated_live_leaves, args.A_total, args.B_total, args.logi_params, args.selection_strength)  # note that when this updates in the middle of the loop over leaves, it'll set lambda values for any children that have so far been added
                        updated_mean_kd = numpy.mean([l.Kd for l in updated_live_leaves if l.Kd != float('inf')])  # NOTE that if mean kd deviates by a huge amount from args.mature_kd (probably due to very low selection strength, causing the cells to rapidly drift away from the target sequence) then in order to avoid the cells dying out you'd need to recalculate A_total here with the current mean kd (but it's unclear what you'd really be simulating then, since adding an aribtrarily large amount of new antigen because the population's getting low isn't very realistic)
                        lambda_update_dbg_str = selection_utils.color('blue', 'x')
                    skip_lambda_n -= 1

                def get_n_children():
                    return numpy.random.poisson(leaf.lambda_ if args.selection else args.lambda_)
                n_children = get_n_children()
                if has_stop_aa(leaf.aa_seq) and n_children > 0:  # shouldn't happen any more (it was a bug when I added selection strength scaling), but let's leave this here just in case
                    print('    %s non-zero children for leaf with stop codon' % selection_utils.color('red', 'error'))
                if current_time == 1 and len(static_live_leaves) == 1:  # if the first leaf draws zero children, keep trying so we don't have to throw out the whole tree and start over
                    itry = 0
                    while n_children == 0:
                        n_children = get_n_children()
                        itry += 1
                        if itry > 10 == 0:
                            print('too many tries to get at least one child, giving up on tree')
                            break

                self.n_unterminated_leaves += n_children - 1
                updated_live_leaves.remove(leaf)  # either it's terminated (no children), or it's got children, either way it's no longer a live leaf
                if n_children == 0:  # kill everyone with no children
                    leaf.terminated = True
                    self.get_ancestor_above_leaf_to_detach(leaf)
                    if len(static_live_leaves) == 1:
                        print('  terminating only leaf in tree (it has no children)')

                if args.debug > 1:
                    n_mutation_list, kd_list = [], []
                for _ in range(n_children):
                    if args.naive_seq2 is not None:  # for paired heavy/light we mutate them separately with their own mutation rate
                        mfos = [self.mutate(get_pair_seq(leaf.nuc_seq, args.pair_bounds, iseq), args.lambda0[iseq]) for iseq in range(len(args.lambda0))]  # NOTE doesn't pass or get aa_seq, but the only result of that should be that self.init_node() has to calculate it
                        mutated_sequence = ''.join(m['nuc_seq'] for m in mfos)
                    else:
                        mfo = self.mutate(leaf.nuc_seq, args.lambda0[0], aa_seq=leaf.aa_seq)
                        if args.debug > 1:
                            n_mutation_list.append(mfo['n_muts'])
                    child = self.init_node(args, mfo['nuc_seq'], current_time, leaf, target_seqs, aa_seq=mfo['aa_seq'], mean_kd=updated_mean_kd)
                    if args.selection and args.debug > 1:
                        kd_list.append(child.Kd)
                    leaf.add_child(child)
                    updated_live_leaves.append(child)
                if args.debug > 1:
                    terminated_dbg_str = selection_utils.color('red', 'x') if n_children == 0 else ' '
                    n_mutation_str_list = [('%d' % n) if n > 0 else '-' for n in n_mutation_list]
                    kd_str_list = ['%.0f' % kd for kd in kd_list]
                    rel_kd_str_list = ['%.2f' % (kd / updated_mean_kd) for kd in kd_list]
                    pre_leaf_str = '' if static_live_leaves.index(leaf) == 0 else ('%12s %3d' % ('', len(updated_live_leaves)))
                    print(('      %s    %4d      %5.2f  %3d  %s%s          %-14s       %-28s      %-28s') % (pre_leaf_str, static_live_leaves.index(leaf), leaf.lambda_, n_children, lambda_update_dbg_str, terminated_dbg_str, ' '.join(n_mutation_str_list), ' '.join(kd_str_list), ' '.join(rel_kd_str_list)))

            if args.selection:
                self.tdist_hists[current_time] = self.get_target_distance_hist(args, updated_live_leaves)
                self.n_mutated_hists[current_time] = scipy.histogram([l.naive_distance for l in updated_live_leaves], bins=list(numpy.arange(-0.5, (max(args.obs_times) if args.obs_times is not None else current_time) + 0.5)))  # can't have more than one mutation per generation

            finished, successful, dbgstr, termstr = self.check_termination(args, current_time, updated_live_leaves)

            if args.debug == 1:
                mintd, meantd = '-', '-'
                minkd, meankd = '-', '-'
                maxl, meanl = '-', '-'  # NOTE don't use <updated_live_leaves> to get lambda values here, since it's either entirely or partly full on unset values from children that were just added
                if args.selection and len(updated_live_leaves) > 0:  # NOTE when you get to here, there are either zero lambda values set in <updated_live_leaves> (if lambdas updated only at the start of the loop) or some intermediate number (if they updated also partway through)
                    tmptdvals = [l.target_distance for l in updated_live_leaves]
                    mintd, meantd = '%2d' % min(tmptdvals), '%3.1f' % numpy.mean(tmptdvals)
                    tmpkdvals = [l.Kd for l in updated_live_leaves if l.Kd != float('inf')]
                    minkd, meankd = [('%5.1f' % v) for v in (min(tmpkdvals), numpy.mean(tmpkdvals))] if len(tmpkdvals) > 0 else (0, 0)
                    if len(updated_lambda_values) > 0:
                        maxl, meanl = [('%4.2f' % v) for v in (max(updated_lambda_values), numpy.mean(updated_lambda_values))]
                print('        %3d       %5d         %s  %s          %s  %s     %4s  %4s      %s' % (current_time, len(updated_live_leaves), mintd, meantd, minkd, meankd, maxl, meanl, dbgstr))

            if finished:
                print(termstr)
                break

            if args.obs_times is not None and len(args.obs_times) > 1 and current_time in args.obs_times:
                self.sample_intermediates(args, current_time, tree)  # note that we don't need to update <updated_live_leaves> in this fcn

        # write a histogram of the hamming distances to target at each generation
        if args.selection and not args.dont_write_hists:
            with open(args.outbase + '_min_aa_target_hdists.p', 'wb') as histfile:
                pickle.dump(self.tdist_hists, histfile)
            with open(args.outbase + '_n_mutated_nuc_hdists.p', 'wb') as histfile:
                pickle.dump(self.n_mutated_hists, histfile)

        stop_leaves = [l for l in updated_live_leaves if has_stop_aa(l.aa_seq)]
        non_stop_leaves = [l for l in updated_live_leaves if not has_stop_aa(l.aa_seq)]
        if len(stop_leaves) > 0:
            print('    %d / %d leaves at final time point have stop codons' % (len(stop_leaves), len(stop_leaves) + len(non_stop_leaves)))

        tree.name = 'naive' if args.observe_common_ancestors else ''  # overwritten below if --observe_common_ancestors is set UPDATE I'm making it so the root has name '' if we're not observing common ancestors (since I don't want to have it in the final annotation), but I don't want to remove the 'naive' if we are observing common ancestors since I'm not sure if it makes a difference
        potential_names, used_names = None, None
        _, potential_names, used_names = selection_utils.choose_new_uid(potential_names, used_names, initial_length=args.uid_str_len, shuffle=True)  # call once (ignoring the returned <uid>) to get the initial length right, and to shuffle them (shuffling is so if we're running multiple events, they have different leaf names, as long as we set the seeds differently)

        if args.obs_times is not None and len(args.obs_times) > 1:  # observe all intermediate sampled nodes
            print('    labeling/observing %d intermediates ' % len(self.intermediate_sampled_nodes))
            for node in self.intermediate_sampled_nodes:
                node.frequency = 1
                uid, potential_names, used_names = selection_utils.choose_new_uid(potential_names, used_names)
                node.name = 'int-' + uid

        observed_leaves = list(non_stop_leaves)  # don't really need a separate list, but it's a little nicer
        if args.n_to_sample is not None and len(observed_leaves) > args.n_to_sample[-1]:  # if there's more leaves than we were asked to sample
            observed_leaves = self.choose_leaves_to_sample(args, observed_leaves, args.n_to_sample[-1])
            print('    sampled %d / %d no-stop leaves at final time (sampling scheme: %s)' % (len(observed_leaves), len(non_stop_leaves), args.leaf_sampling_scheme))

        for leaf in observed_leaves:
            leaf.frequency = 1
            uid, potential_names, used_names = selection_utils.choose_new_uid(potential_names, used_names)
            leaf.name = 'leaf-' + uid

        for leaf in set(updated_live_leaves) - set(observed_leaves):
            self.get_ancestor_above_leaf_to_detach(leaf)

        if args.selection:
            self.sampled_tdist_hists[current_time] = self.get_target_distance_hist(args, observed_leaves)  # NOTE this doesn't include nodes added from --observe_common_ancestors or --observe_all_ancestors
            if len(self.sampled_tdist_hists) > 0 and not args.dont_write_hists:
                with open(args.outbase + '_sampled_min_aa_target_hdists.p', 'wb') as histfile:
                    pickle.dump(self.sampled_tdist_hists, histfile)

        print('    detaching %d nodes' % len(self.nodes_to_detach))
        for node in self.nodes_to_detach:
            node.detach()

        # note: don't need this anymore now that we have get_ancestor_above_leaf_to_detach(), but I don't want to delete it yet
        # # prune away lineages that have zero total observation frequency
        # n_pruned_lineages, n_pruned_nodes = 0, 0
        # start = time.time()
        # detached_descendents = set()
        # for node in tree.iter_descendants():  # NOTE this is kinda slow, and it might (might!) be faster (UPDATE: is definitely faster) to propagate the information upward when we set the observed nodes to start with (rather than looping over descendents here)), but it's quite a bit faster than it used to be already so not going to mess around further a.t.m.
        #     if node in detached_descendents:
        #         # detached_descendents.remove(node)  # don't need it in there any more, but it isn't any faster to remove it
        #         continue
        #     if any(child.frequency > 0 for child in node.traverse()):  # if all children of <node> have zero observation frequency, detach <node> (only difference between traverse() and iter_descendants() seems to be that traverse() includes the node on which you're calling it, while iter_descendants() doesn't)
        #         continue
        #     node.detach()
        #     for child in node.iter_descendants():  # avoid checking <node>'s children
        #         detached_descendents.add(child)
        #         n_pruned_nodes += 1
        #     n_pruned_lineages += 1
        # print('    removed %d nodes in %d unobserved lineages (%.1fs)' % (n_pruned_nodes, n_pruned_lineages, time.time()-start))

        # remove unobserved unifurcations and nodes that are distance zero from their parents
        n_removed = 0
        for node in tree.iter_descendants():
            parent = node.up
            if node.frequency == 0 and (len(node.children) == 1 or node.dist == 0):  # NOTE this allows leaves that are distance zero from their parents, since leaves we get here are observed
                node.delete(prevent_nondicotomic=False)  # seems like this should instead use the preserve_branch_length=True option so we don't need the next line, but I don't want to change it
                if node.dist > 0:
                    for child in node.children:
                        child.dist = hamming_distance(child.nuc_seq, parent.nuc_seq)
                n_removed += 1
        print('    removed %d unobserved unifurcations + degenerate nodes' % n_removed)

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
            print('    added %d common ancestor nodes (%.1fs)' % (n_observed_ancestors, time.time()-start))  # oh, wait, maybe this doesn't take any real time any more? i thought it used to involve more iteration/traversing

        # neutral collapse will fail if there's backmutations (?) [preserving old comment]
        treename = 'GCsim %s' % ('selection' if args.selection else 'neutral')
        collapsed_tree = CollapsedTree(tree, treename, allow_repeats=args.selection)

        tree.ladderize()

        return tree, collapsed_tree, successful

# ----------------------------------------------------------------------------------------
def make_plots(args, tree, collapsed_tree):
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

    tree.render(args.outbase + '_lineage_tree.svg', tree_style=ts)

    # Render collapsed tree,
    # create an id-wise colormap
    # NOTE: node.name can be a set
    if args.plotAA and args.selection:
        colormap = {node.name:colors[node.aa_seq] for node in collapsed_tree.tree.traverse()}
    else:
        colormap = {node.name:colors[node.nuc_seq] for node in collapsed_tree.tree.traverse()}
    collapsed_tree.render(args.outbase+'_collapsed_tree.svg', idlabel=args.idlabel, colormap=colormap)
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
        collapsed_tree.render(args.outbase+'_collapsed_runstat_color_tree.svg', idlabel=args.idlabel, colormap=colormap)
        if not args.dont_write_hists:
            with open(args.outbase + '_min_aa_target_hdists.p', 'rb') as fh:
                tdist_hists = pickle.load(fh)
                selection_utils.plot_runstats(tdist_hists, args.outbase, colors)

# ----------------------------------------------------------------------------------------
def run_simulation(args):
    start = time.time()
    mutation_model = MutationModel(args)
    itry = 0
    while itry < args.n_tries:
        if itry > 0:
            print('  itry %d: retrying tree simulation' % itry)
        tree, collapsed_tree, successful = mutation_model.simulate(args)
        if successful:
            break
        itry += 1
    if not successful:
        raise Exception('didn\'t succeed after %d tries' % args.n_tries)

    # write observed sequences to fasta file(s)
    if args.naive_seq2 is not None:
        fhandles = [open('%s_seq%d.fasta' % (args.outbase, iseq + 1), 'w') for iseq in range(2)]
        for iseq, fh in enumerate(fhandles):
            fh.write('>%s\n%s\n' % (tree.name, get_pair_seq(args.naive_tseq.nuc, args.pair_bounds, iseq)))  # NOTE not trimming off the n_pads_added stuff, it's too hard with the multiple sequences concatenated
        for node in [n for n in tree.iter_descendants() if n.frequency != 0]:  # NOTE doesn't iterate over root node
            for iseq, fh in enumerate(fhandles):
                fh.write('>%s\n%s\n' % (node.name, get_pair_seq(node.nuc_seq, args.pair_bounds, iseq)))
    else:
        with open('%s.fasta' % args.outbase, 'w') as fh:
            if args.observe_common_ancestors:
                fh.write('>%s\n%s\n' % (tree.name, args.naive_tseq.nuc))  # [:len(args.naive_tseq.nuc) - args.n_pads_added]))
            for node in [n for n in tree.iter_descendants() if n.frequency != 0]:  # NOTE doesn't iterate over root node
                fh.write('>%s\n%s\n' % (node.name, node.nuc_seq))  # [:len(node.nuc_seq) - args.n_pads_added]))

    # write some summary statistics
    frequency, distance_from_naive, degree = zip(*[(node.frequency,
                                                    node.naive_distance,
                                                    sum(hamming_distance(node.nuc_seq, node2.nuc_seq) == 1 for node2 in collapsed_tree.tree.traverse() if node2.frequency and node2 is not node))
                                                   for node in collapsed_tree.tree.traverse() if node.frequency])
    stats = pd.DataFrame({'genotype abundance':frequency,
                          'Hamming distance to root genotype':distance_from_naive,
                          'Hamming neighbor genotypes':degree})
    stats.to_csv(args.outbase+'_stats.tsv', sep='\t', index=False)

    print('  observed %d simulated sequences%s' % (sum(node.frequency for node in collapsed_tree.tree.traverse()), '' if args.no_context else ' (with context dependence)'))

    if not args.no_plot:  # put this before we pickledump them, so the style gets written to the pickle files
        make_plots(args, tree, collapsed_tree)
    with open(args.outbase+'_lineage_tree.p', 'wb') as f:
        pickle.dump(tree, f)
    collapsed_tree.write(args.outbase + '_collapsed_tree.p')
    if args.selection:
        collapsed_tree.write( args.outbase+'_collapsed_runstat_color_tree.p')

    print('  total time %.1fs' % (time.time() - start))

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
    parser.add_argument('--n_tries', type=int, default=1, help='If the tree terminates before the specified stopping criteria are met, set larger than 1 to keep trying (effectively, with different --random_seed).')
    parser.add_argument('--lambda', dest='lambda_', type=float, default=0.9, help='Poisson branching parameter to use if selection is turned off.')
    parser.add_argument('--lambda0', type=float, nargs='*', help='Baseline sequence mutation rate(s): first value corresponds to --naive_seq, and optionally the second to --naive_seq2. If only one rate is provided for two sequences, this rate is used for both. If not set, the default is set below')
    parser.add_argument('--target_sequence_lambda0', type=float, default=0.1, help='baseline mutation rate used for generating target sequences (you shouldn\'t need to change this)')
    parser.add_argument('--selection_strength', type=float, default=1., help='Value in [0, 1] specifying the relative strength of selection, i.e. the extent to which fitness (in the form of the lambda value for each cell\'s poisson distribution from which its N offspring is drawn) is determined by affinity (strength of 1) vs by chance (strength of 0)')
    parser.add_argument('--n_to_sample', type=int, nargs='+', help='Number of cells to sample from the final generation (if one value), or at each generation specified in --obs-times (if same length as --obs_times, and both are set). If --obs_times is set and has more than one value, but --n_to_sample is length one, this same value is applied to each time in --obs_times.')
    parser.add_argument('--kill_sampled_intermediates', action='store_true', help='kill intermediate sequences as they are sampled')
    parser.add_argument('--observe_common_ancestors', action='store_true', help='If set, after deciding which nodes to observe (write to file) according to other options, we then also select the most recent common ancestor for every pair of those nodes (the idea is that this gets you the nodes that you would reconstruct with a phylogenetic program). NOTE histograms written to disk currently don\'t include these.')
    parser.add_argument('--carry_cap', type=int, default=1000, help='The carrying capacity of the simulation with selection. This number affects the fixation time of a new mutation.'
                        'Fixation time is approx. log2(carry_cap), e.g. log2(1000) ~= 10.')
    parser.add_argument('--target_count', type=int, default=10, help='The number of target sequences to generate.')
    parser.add_argument('--target_distance', type=int, default=10, help='Desired distance (using --metric_for_target_distance) between the naive sequence and the target sequences.')
    parser.add_argument('--n_target_clusters', type=int, help='If set, divide the --target_count target sequences into --target_count / --n_target_clusters "clusters" of target sequences, where each cluster consists of one "main" sequence separated from the naive by --target_distance, surrounded by the others in the cluster at radius --target_cluster_distance. If you set numbers that aren\'t evenly divisible, then the clusters won\'t all be the same size, but the total number of targets will always be --target_count')
    parser.add_argument('--target_cluster_distance', type=int, default=1, help='See --target_cluster_count')
    parser.add_argument('--min_target_distance', type=int, help='If set, the target distance used to calculate affinity can never fall below this value, even if the cell\'s sequence is closer than this to a target sequence. This makes it so cells can bounce around within this threshold of distance, rather than being sucked into exactly the target sequence.')
    parser.add_argument('--metric_for_target_distance', default='aa', choices=['aa', 'nuc', 'aa-sim-ascii', 'aa-sim-blosum'], help='Metric to use to calculate the distance to each target sequence (aa: use amino acid distance, i.e. only non-synonymous mutations count, nuc: use nucleotide distance, aa-sim: amino acid distance, but where different pairs of amino acids are different distances apart, with either ascii-code-based or blosum-based distances).')
    parser.add_argument('--paratope_positions', default='all', choices=['all', 'cdrs'], help='Positions in each sequence that should be considered as part of the paratope, i.e. that count toward the target distance (non-paratope positions are ignored for purposes of the target distance). \'all\' uses all positions, \'cdrs\' uses half the positions (not actually the cdr positions a.t.m.).')
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
    parser.add_argument('--no_plot', action='store_true', help='don\'t write any plots (they\'re pretty slow), although we stil write some .p historgrams (see --dont_write_hists).')
    parser.add_argument('--dont_write_hists', action='store_true', help='don\'t write any of the .p histograms (they\'re much larger than the fasta + tree files that we really care about)')
    parser.add_argument('--debug', type=int, default=0, choices=[0, 1, 2], help='Debug verbosity level.')
    parser.add_argument('--outbase', default='GCsimulator_out', help='Output file base name')
    parser.add_argument('--idlabel', action='store_true', help='Flag for labeling the sequence ids of the nodes in the output tree images, also write associated fasta alignment if True')
    parser.add_argument('--random_seed', type=int, help='for random number generator')
    parser.add_argument('--pair_bounds', help='for internal use only')
    parser.add_argument('--leaf_sampling_scheme', default='uniform-random', choices=['uniform-random', 'affinity-biased', 'high-affinity'], help='When selecting cells to observe, we can either sample entirely randomly (\'uniform-random\', default), randomly sample with each cell\'s weight 1/Kd (\'affinity-biased\'), or sort cells by Kd and choose precisely the highest-affinity cells (\'high-affinity\').')
    parser.add_argument('--verbose', action='store_true', help='DEPRECATED use --debug')
    parser.add_argument('--n_to_downsample', type=int, nargs='+', help='DEPRECATED use --n_to_sample')
    parser.add_argument('--uid_str_len', type=int, default=4, help='Number of random lowercase letters to use to construct each node\'s names.')
    # parser.add_argument('--n_pads_added', type=int, default=0, help='INTERNAL USE ONLY')

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
    # if len(args.naive_seq) % 3 != 0:  # this lets you remove the extra lines in GCutils.local_translate(), which is faster, but then the N pads get passed to partis, which confuses things (especially since they can get mutated)
    #     print('  note: padding right side of --naive_seq to multiple of three')
    #     n_pads_added = 3 - (len(args.naive_seq) % 3)
    #     args.naive_seq += 'N' * n_pads_added
    #     args.n_pads_added = n_pads_added
    args.naive_tseq = TranslatedSeq(args, args.naive_seq)
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
        # print('%s not padding naive_seq2 to length multiple of 3' % selection_utils.color('red', 'warning:'))
        if len(args.lambda0) == 1:  # Use the same mutation rate on both sequences
            args.lambda0 = [args.lambda0[0], args.lambda0[0]]
        elif len(args.lambda0) != 2:
            raise Exception('--lambda0 (set to \'%s\') has to have either two values (if --naive_seq2 is set) or one value (if it isn\'t).' % args.lambda0)
        if len(args.naive_tseq.nuc) % 3 != 0:  # have to pad first one out to a full codon so we don't think there's a bunch of stop codons in the second sequence
            args.naive_tseq.nuc += 'N' * (3 - len(args.naive_tseq.nuc) % 3)
        args.pair_bounds = ((0, len(args.naive_tseq.nuc)), (len(args.naive_tseq.nuc), len(args.naive_tseq.nuc + args.naive_seq2)))  # bounds to allow mashing the two sequences toegether as one string
        args.naive_tseq = TranslatedSeq(args, args.naive_tseq.nuc + args.naive_seq2.upper())  # merge the two seqeunces to simplify future dealing with the pair:
        if has_stop(args.naive_tseq.nuc):
            raise Exception('stop codon in --naive_seq2 (this isn\'t necessarily otherwise forbidden, but it\'ll quickly end you in a thicket of infinite loops, so should be corrected).')

    if args.selection:
        assert args.target_distance > 0
        if args.min_target_distance is not None and args.min_target_distance >= args.target_distance:
            raise Exception('--min_target_distance %d has to be less than --target_distance %d' % (args.min_target_distance, args.target_distance))
        assert args.B_total >= args.f_full  # the fully activating fraction on BA must be possible to reach within B_total
        args.A_total = selection_utils.find_A_total(args.carry_cap, args.B_total, args.f_full, args.mature_kd, args.U)  # find the total amount of A necessary for sustaining the specified carrying capacity
        args.logi_params = selection_utils.find_logistic_params(args.f_full, args.U)  # calculate the parameters for the logistic function

    run_simulation(args)

# ----------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
