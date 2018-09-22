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

from GCutils import hamming_distance, has_stop, translate, CollapsedTree
import selection_utils

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
    def init_node(self, sequence, time, distance, selection=False):
        node = TreeNode()
        node.add_feature('sequence', sequence)
        node.add_feature('time', time)
        node.dist = distance  # not sure why this isn't using add_feature(), but I don't want to change it
        node.add_feature('terminated', False)  # set if it's dead (only set if it draws zero children, or if --kill_sampled_intermediates is set and it's sampled at an intermediate time point)
        node.add_feature('intermediate_sampled', False)  # set if it's sampled at an intermediate time point
        node.add_feature('frequency', 0)  # observation frequency, seems to always be either 1 or 0 (set in set_observation_frequencies_and_names())
        if selection:
            node.add_feature('lambda_', None)  # set in selection_utils.update_lambda_values()
            node.add_feature('AAseq', None)  # set in MutationModel.simulate()
            node.add_feature('Kd', None)  # same
            node.add_feature('target_distance', None)  # same
        return node

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
    def make_target_sequence(self, naive_seq, aa_naive_seq, n_muts, lambda0, n_max_tries=100):
        '''
        Make a single target sequence with <n_muts> hamming distance from <naive_seq> (in amino acid space)
        '''
        assert not has_stop(naive_seq)  # already checked during argumetent parsing, but we really want to make sure since the loop will blow up if there's a stop to start with
        itry = 0
        while itry < n_max_tries:
            dist = None
            while dist is None or dist < n_muts:
                mut_seq = self.mutate(naive_seq if dist is None else mut_seq, lambda0)
                aa_mut = translate(mut_seq)
                dist = hamming_distance(aa_naive_seq, aa_mut)
                if dist == n_muts and '*' not in aa_mut:  # Stop codon cannot be part of the return
                    return aa_mut
            itry += 1

        raise RuntimeError('fell through after trying %d times to make a target sequence' % n_max_tries)

    # ----------------------------------------------------------------------------------------
    def set_observation_frequencies_and_names(self, args, tree, current_time, final_leaves):
        tree.get_tree_root().name = 'naive'  # doesn't seem to get written properly

        potential_names, used_names = None, None

        if args.obs_times is not None and len(args.obs_times) > 1:  # observe all intermediate sampled nodes
            for node in [l for l in tree.iter_descendants() if l.intermediate_sampled]:
                node.frequency = 1
                uid, potential_names, used_names = selection_utils.choose_new_uid(potential_names, used_names, initial_length=3)
                node.name = 'int-' + uid

        if args.n_to_downsample is not None and len(final_leaves) > args.n_to_downsample[-1]:  # if we were asked to downsample, and if there's enough leaves to do so
            final_leaves = random.sample(final_leaves, args.n_to_downsample[-1])

        for leaf in final_leaves:
            leaf.frequency = 1
            uid, potential_names, used_names = selection_utils.choose_new_uid(potential_names, used_names, initial_length=3)
            leaf.name = 'leaf-' + uid

        if args.observe_common_ancestors:
            observed_nodes = [n for n in tree.iter_descendants() if n.frequency == 1]
            for node_1, node_2 in itertools.combinations(observed_nodes, 2):
                mrca = node_1.get_common_ancestor(node_2)
                # print('    %s, %s:  %s' % (node_1.name, node_2.name, mrca.name))
                mrca.frequency = 1
                uid, potential_names, used_names = selection_utils.choose_new_uid(potential_names, used_names, initial_length=3)
                mrca.name = 'mrca-' + uid

    # ----------------------------------------------------------------------------------------
    def simulate(self, args):
        '''
        Simulate a poisson branching process with mutation introduced
        by the chosen mutation model e.g. motif or uniform.
        Can either simulate under a neutral model without selection,
        or using an affinity muturation inspired model for selection.
        '''

        tree = self.init_node(args.naive_seq, time=0, distance=0, selection=args.selection)

        if args.selection:
            hd_generation = list()  # Collect an array of the counts of each hamming distance (to a target) at each time step

            aa_naive_seq = translate(tree.sequence)

            print('    making %d target sequences' % args.target_count)
            targetAAseqs = [self.make_target_sequence(args.naive_seq, aa_naive_seq, args.target_distance, args.target_sequence_lambda0) for i in range(args.target_count)]

            assert len(set([len(aa_naive_seq)] + [len(t) for t in targetAAseqs])) == 1  # targets and naive seq are same length
            assert len(set([args.target_distance] + [hamming_distance(aa_naive_seq, t) for t in targetAAseqs])) == 1  # all targets are the right distance from the naive

            def hd2affy(hd):  # affinity is an exponential function of hamming distance
                return(args.mature_affy + hd**args.k_exp * (args.naive_affy - args.mature_affy) / args.target_distance**args.k_exp)

            tree.AAseq = str(aa_naive_seq)
            tree.Kd = selection_utils.calc_Kd(tree.AAseq, targetAAseqs, hd2affy)
            tree.target_distance = args.target_distance

        print('    starting %d generations' % max(args.obs_times))
        if args.verbose:
            print('       gen   live leaves')
            print('             before/after   ileaf   n children     n mutations          Kd   (%s: updated lambdas)' % selection_utils.color('blue', 'x'))
        current_time = 0
        n_unterminated_leaves = 1
        hd_distrib = []
        while True:
            if n_unterminated_leaves <= 0:  # if everybody's dead (probably can't actually go less than zero, but not sure)
                break
            if args.n_final_seqs is not None and n_unterminated_leaves >= args.n_final_seqs:  # if we've got as many sequences as we were asked for
                break
            if args.obs_times is not None and current_time >= max(args.obs_times):  # if we've done as many generations as we were told to
                break
            if args.stop_dist is not None and current_time > 0 and args.stop_dist < min(hd_distrib):  # if the leaves have gotten close enough to the target sequences
                break

            # sample any requested intermediate time points (from *last* generation, since haven't yet incremented current_time)
            if args.obs_times is not None and len(args.obs_times) > 1 and current_time in args.obs_times:
                assert len(args.obs_times) == len(args.n_to_downsample)
                n_to_sample = args.n_to_downsample[args.obs_times.index(current_time)]
                live_nostop_leaves = [l for l in tree.iter_leaves() if not l.terminated and not has_stop(l.sequence)]
                if len(live_nostop_leaves) < n_to_sample:
                    raise RuntimeError('tried to sample %d leaves at intermediate timepoint %d, but tree only has %d live leaves without stops (try a later generation or larger carrying capacity).' % (n_to_sample, current_time, len(live_nostop_leaves)))
                for leaf in random.sample(live_nostop_leaves, n_to_sample):
                    leaf.intermediate_sampled = True
                    if args.kill_sampled_intermediates:
                        leaf.terminated = True
                        n_unterminated_leaves -= 1
                print('                  sampled %d (of %d live and stop-free) intermediate leaves (%s) at time %d (but time is about to increment to %d)' % (n_to_sample, len(live_nostop_leaves), 'killing each of them' if args.kill_sampled_intermediates else 'leaving them alive', current_time, current_time + 1))

            current_time += 1

            skip_lambda_n = 0  # index keeping track of how many leaves since we last updated all the lambdas
            live_leaves = [l for l in tree.iter_leaves() if not l.terminated]  # NOTE this is out of date as soon as we've added any children in the loop, or killed anybody with no children
            updated_live_leaves = [l for l in live_leaves]  # but this one, we keep updating (so we don't have to call iter_leaves() so much, which was taking quite a bit of time)
            random.shuffle(live_leaves)
            if args.verbose:
                print('      %3d    %3d' % (current_time, len(live_leaves)), end='\n' if len(live_leaves) == 0 else '')  # NOTE these are live leaves *after* the intermediate sampling above
            for leaf in live_leaves:
                if args.selection:
                    lambda_update_dbg_str = ' '
                    if skip_lambda_n == 0:  # time to update the lambdas for every leaf
                        skip_lambda_n = args.skip_update + 1  # reset it so we don't update again until we've done <args.skip_update> more leaves (+ 1 is so that if args.skip_update is 0 we don't skip at all, i.e. args.skip_update is the number of leaves skipped, *not* the number of leaves *between* updates)
                        tree = selection_utils.update_lambda_values(tree, updated_live_leaves, targetAAseqs, hd2affy, args.A_total, args.B_total, args.Lp)
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

                n_unterminated_leaves += n_children - 1  # <-- Getting 1, is equal to staying alive
                if n_children == 0:
                    leaf.terminated = True
                    updated_live_leaves.remove(leaf)
                    if len(live_leaves) == 1:
                        print('  terminating only leaf with no children')

                if args.verbose:
                    n_mutation_list, kd_list = [], []
                for _ in range(n_children):
                    if args.naive_seq2 is not None:  # for paired heavy/light we mutate them separately with their own mutation rate
                        mutated_sequence1 = self.mutate(get_seq_from_pair(leaf.sequence, args.pair_bounds, iseq=0), args.lambda0[0])
                        mutated_sequence2 = self.mutate(get_seq_from_pair(leaf.sequence, args.pair_bounds, iseq=1), args.lambda0[1])
                        mutated_sequence = mutated_sequence1 + mutated_sequence2
                    else:
                        mutated_sequence, n_muts = self.mutate(leaf.sequence, args.lambda0[0], return_n_mutations=True)
                        if args.verbose:
                            n_mutation_list.append(n_muts)
                    child = self.init_node(mutated_sequence, current_time, distance=sum(x!=y for x,y in zip(mutated_sequence, leaf.sequence)), selection=args.selection)
                    if args.selection:
                        child.AAseq = str(translate(child.sequence))
                        child.Kd = selection_utils.calc_Kd(child.AAseq, targetAAseqs, hd2affy)
                        child.target_distance = min([hamming_distance(child.AAseq, taa) for taa in targetAAseqs])
                        if args.verbose:
                            kd_list.append(child.Kd)
                    leaf.add_child(child)
                    updated_live_leaves.append(child)
                    if leaf in updated_live_leaves:  # leaf isn't a leaf any more, since now it has children
                        updated_live_leaves.remove(leaf)
                if args.verbose:
                    n_mutation_str_list = [('%d' % n) if n > 0 else '-' for n in n_mutation_list]
                    kd_str_list = ['%.0f' % kd for kd in kd_list]
                    pre_leaf_str = '' if live_leaves.index(leaf) == 0 else ('%12s %3d' % ('', len(updated_live_leaves)))
                    print(('      %s      %4d   %3d  %s          %-14s       %-28s') % (pre_leaf_str, live_leaves.index(leaf), n_children, lambda_update_dbg_str, ' '.join(n_mutation_str_list), ' '.join(kd_str_list)))
            if args.selection:
                hd_distrib = [min([hamming_distance(tn.AAseq, ta) for ta in targetAAseqs]) for tn in updated_live_leaves]  # list, for each live leaf, of the smallest distance to any target
                n_bins = args.target_distance * 10 if args.target_distance > 0 else 10
                hist = scipy.histogram(hd_distrib, bins=list(range(n_bins)))
                hd_generation.append(hist)
                # if args.verbose and hd_distrib:
                #     print('            total population %-4d    majority distance to target %-3d' % (sum(hist[0]), scipy.argmax(hist[0])))
            # if current_time > 5:
            #     assert False

        # write a histogram of the hamming distances to target at each generation
        if args.selection:
            with open(args.outbase + '_selection_runstats.p', 'wb') as f:
                pickle.dump(hd_generation, f)

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

        stop_leaves = [l for l in tree.iter_leaves() if l.time == current_time and has_stop(l.sequence)]
        non_stop_leaves = [l for l in tree.iter_leaves() if l.time == current_time and not has_stop(l.sequence)]  # non-stop leaves
        if len(stop_leaves) > 0:
            print('    %d / %d leaves at final time point have stop codons' % (len(stop_leaves), len(stop_leaves) + len(non_stop_leaves)))

        self.set_observation_frequencies_and_names(args, tree, current_time, non_stop_leaves)

        # prune away lineages that have zero total observation frequency
        for node in tree.iter_descendants():
            if sum(child.frequency for child in node.traverse()) == 0:  # if all children of <node> have zero observation frequency, detach <node> (only difference between traverse() and iter_descendants() seems to be that traverse() includes the node on which you're calling it, while iter_descendants() doesn't)
                node.detach()

        # NOTE duplicates code in CollapsedTree
        # remove unobserved unifurcations
        for node in tree.iter_descendants():
            parent = node.up
            if node.frequency == 0 and len(node.children) == 1:
                node.delete(prevent_nondicotomic=False)
                node.children[0].dist = hamming_distance(node.children[0].sequence, parent.sequence)

        if args.selection:
            collapsed_tree = CollapsedTree(tree=tree, name='GCsim selection', collapse_syn=False, allow_repeats=True)
        else:
            collapsed_tree = CollapsedTree(tree=tree, name='GCsim neutral')  # <-- This will fail if backmutations
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
            fh.write('>naive\n%s\n' % get_seq_from_pair(args.naive_seq, args.pair_bounds, iseq=iseq))
        for node in [n for n in tree.iter_descendants() if n.frequency != 0]:
            for iseq, fh in enumerate(fhandles):
                fh.write('>%s\n%s\n' % (node.name, get_seq_from_pair(node.sequence, args.pair_bounds, iseq=iseq)))
    else:
        with open('%s.fasta' % args.outbase, 'w') as fh:
            fh.write('>naive\n%s\n' % args.naive_seq)
            for node in [n for n in tree.iter_descendants() if n.frequency != 0]:
                fh.write('>%s\n%s\n' % (node.name, node.sequence))

    # write some observable simulation stats
    frequency, distance_from_naive, degree = zip(*[(node.frequency,
                                                    hamming_distance(node.sequence, args.naive_seq),
                                                    sum(hamming_distance(node.sequence, node2.sequence) == 1 for node2 in collapsed_tree.tree.traverse() if node2.frequency and node2 is not node))
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
        colors[tree.AAseq] = 'gray'
    else:
        colors[tree.sequence] = 'gray'

    for n in tree.traverse():
        nstyle = NodeStyle()
        nstyle["size"] = 10
        if args.plotAA:
            if n.AAseq not in colors:
                colors[n.AAseq] = next(palette)
            nstyle['fgcolor'] = colors[n.AAseq]
        else:
            if n.sequence not in colors:
                colors[n.sequence] = next(palette)
            nstyle['fgcolor'] = colors[n.sequence]
        n.set_style(nstyle)

    # Render and pickle lineage tree:
    tree.render(args.outbase+'_lineage_tree.svg', tree_style=ts)
    with open(args.outbase+'_lineage_tree.p', 'wb') as f:
        pickle.dump(tree, f)

    # Render collapsed tree,
    # create an id-wise colormap
    # NOTE: node.name can be a set
    if args.plotAA and args.selection:
        colormap = {node.name:colors[node.AAseq] for node in collapsed_tree.tree.traverse()}
    else:
        colormap = {node.name:colors[node.sequence] for node in collapsed_tree.tree.traverse()}
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
        colors = {i: next(palette) for i in range(int(len(args.naive_seq) // 3))}
        # The minimum distance to the target is colored:
        colormap = {node.name:colors[node.target_distance] for node in collapsed_tree.tree.traverse()}
        collapsed_tree.write( args.outbase+'_collapsed_runstat_color_tree.p')
        collapsed_tree.render(args.outbase+'_collapsed_runstat_color_tree.svg',
                              idlabel=args.idlabel,
                              colormap=colormap)
        # Write a file with the selection run stats. These are also plotted:
        with open(args.outbase + '_selection_runstats.p', 'rb') as fh:
            runstats = pickle.load(fh)
            selection_utils.plot_runstats(runstats, args.outbase, colors)


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

    parser.add_argument('--naive_seq', default='GGACCTAGCCTCGTGAAACCTTCTCAGACTCTGTCCCTCACCTGTTCTGTCACTGGCGACTCCATCACCAGTGGTTACTGGAACTGGATCCGGAAATTCCCAGGGAATAAACTTGAGTACATGGGGTACATAAGCTACAGTGGTAGCACTTACTACAATCCATCTCTCAAAAGTCGAATCTCCATCACTCGAGACACATCCAAGAACCAGTACTACCTGCAGTTGAATTCTGTGACTACTGAGGACACAGCCACATATTACTGT', help='Initial naive nucleotide sequence.')
    parser.add_argument('--naive_seq_file', default=None, help='Path to fasta file containing initial naive sequences from which do draw at random.')
    parser.add_argument('--mutability_file', default=file_dir+'/../motifs/Mutability_S5F.csv', help='Path to mutability model file.')
    parser.add_argument('--substitution_file', default=file_dir+'/../motifs/Substitution_S5F.csv', help='Path to substitution model file.')
    parser.add_argument('--no_context', action='store_true', help='Disable context dependence, i.e. use a uniform mutability and substitution.')
    parser.add_argument('--selection', action='store_true', help='If set, simulate with selection (otherwise neutral). Requires that you set --obs_times, and therefore that you *not* set --n_final_seqs.')
    parser.add_argument('--n_final_seqs', type=int, default=None, help='If set, simulation stops when we\'ve reached this number of sequences (other stopping criteria: --stop_dist and --obs_times). Because sequences with stop codons are subsequently removed, and because more than on sequence is added per iteration, though we don\'t necessarily output this many. (If --n_to_downsample is also set, then we simulate until we have --n_final_seqs, then downsample to --n_to_downsample).')
    parser.add_argument('--obs_times', type=int, nargs='+', default=None, help='If set, simulation stops when we\'ve reached this many generations. If more than one value is specified, the largest value is the final observation time (and stopping criterion), and earlier values are used as additional, intermediate sampling times (other stopping criteria: --n_final_seqs, --stop_dist)')
    parser.add_argument('--stop_dist', type=int, default=None, help='If set, simulation stops when any simulated sequence is closer than this hamming distance to any of the target sequences (other stopping criteria: --n_final_seqs, --obs_times).')
    parser.add_argument('--lambda', dest='lambda_', type=float, default=0.9, help='Poisson branching parameter')
    parser.add_argument('--lambda0', type=float, default=None, nargs='*', help='Baseline sequence mutation rate(s): first value corresponds to --naive_seq, and optionally the second to --naive_seq2. If only one rate is provided for two sequences, this rate is used for both. If not set, the default is set below')
    parser.add_argument('--target_sequence_lambda0', type=float, default=0.1, help='baseline mutation rate used for generating target sequences (you shouldn\'t need to change this)')
    parser.add_argument('--n_to_downsample', type=int, nargs='+', default=None, help='Number of cells sampled during each sampling step. If one value is specified, this same value is applied to each time in --obs_times; whereas if more than one value is specified, each is applied to the corresponding value in --obs_times.')
    parser.add_argument('--kill_sampled_intermediates', action='store_true', help='kill intermediate sequences as they are sampled')
    parser.add_argument('--observe_common_ancestors', action='store_true', help='If set, after deciding which nodes to observe (write to file) according to other options, we then also select the most recent common ancestor for every pair of those nodes.')
    parser.add_argument('--carry_cap', type=int, default=1000, help='The carrying capacity of the simulation with selection. This number affects the fixation time of a new mutation.'
                        'Fixation time is approx. log2(carry_cap), e.g. log2(1000) ~= 10.')
    parser.add_argument('--target_count', type=int, default=10, help='The number of target sequences to generate.')
    parser.add_argument('--target_distance', type=int, default=10, help='Desired distance (number of non-synonymous mutations) between the naive sequence and the target sequences.')
    parser.add_argument('--naive_seq2', default=None, help='Second seed naive nucleotide sequence. For simulating heavy/light chain co-evolution.')
    parser.add_argument('--naive_affy', type=float, default=100, help='Affinity of the naive sequence in nano molar.')
    parser.add_argument('--mature_affy', type=float, default=1, help='Affinity of the mature sequences in nano molar.')
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
    parser.add_argument('--k_exp', type=float, default=2, help='The exponent in the function to map hamming distance to affinity. '
                        'It is recommended to keep this as the default.')
    parser.add_argument('--plotAA', action='store_true', help='Plot trees with collapsing and coloring on amino acid level.')
    parser.add_argument('--verbose', action='store_true', help='Print progress during simulation. Mostly useful for simulation with selection since this can take a while.')
    parser.add_argument('--outbase', default='GCsimulator_out', help='Output file base name')
    parser.add_argument('--idlabel', action='store_true', help='Flag for labeling the sequence ids of the nodes in the output tree images, also write associated fasta alignment if True')
    parser.add_argument('--random_seed', type=int, help='for random number generator')
    parser.add_argument('--pair_bounds', help='for internal use only')

    args = parser.parse_args()
    if args.random_seed is not None:
        numpy.random.seed(args.random_seed)
        random.seed(args.random_seed)
    if args.no_context:
        args.mutability_file = None
        args.substitution_file = None
    if args.lambda0 is None:
        args.lambda0 = [max([1, int(.01*len(args.naive_seq))])]
    if [args.naive_seq, args.naive_seq_file].count(None) != 1:
        raise Exception('exactly one of --naive_seq and --naive_seq_file must be set')
    if args.naive_seq_file is not None:
        from Bio import SeqIO
        records = list(SeqIO.parse(args.naive_seq_file, "fasta"))
        random.shuffle(records)
        args.naive_seq = str(records[0].seq).upper()
    if args.naive_seq is not None:
        args.naive_seq = args.naive_seq.upper()
    if has_stop(args.naive_seq):
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
            raise Exception('--n_to_downsample has to either be length one, or has to be the same length as --obs_times')
    if args.obs_times is not None:  # sort the observation times
        if args.obs_times != sorted(args.obs_times):
            raise Exception('--obs_times must be sorted (we could sort them here, but then you might think you didn\'t need to worry about the order of --n_to_downsample being the same)')
            # args.obs_times = sorted(args.obs_times)

    if args.naive_seq2 is not None:
        if len(args.lambda0) == 1:  # Use the same mutation rate on both sequences
            args.lambda0 = [args.lambda0[0], args.lambda0[0]]
        elif len(args.lambda0) != 2:
            raise Exception('--lambda0 (set to \'%s\') has to have either two values (if --naive_seq2 is set) or one value (if it isn\'t).' % args.lambda0)
        if len(args.naive_seq) % 3 != 0:  # have to pad first one out to a full codon so we don't think there's a bunch of stop codons in the second sequence
            args.naive_seq += 'N' * (3 - len(args.naive_seq) % 3)
        args.pair_bounds = ((0, len(args.naive_seq)), (len(args.naive_seq), len(args.naive_seq + args.naive_seq2)))  # bounds to allow mashing the two sequences toegether as one string
        args.naive_seq += args.naive_seq2.upper()  # merge the two seqeunces to simplify future dealing with the pair:
        if has_stop(args.naive_seq):
            raise Exception('stop codon in --naive_seq2 (this isn\'t necessarily otherwise forbidden, but it\'ll quickly end you in a thicket of infinite loops, so should be corrected).')

    if args.selection:
        assert args.target_distance > 0
        assert args.B_total >= args.f_full  # the fully activating fraction on BA must be possible to reach within B_total
        # find the total amount of A necessary for sustaining the specified carrying capacity
        args.A_total = selection_utils.find_A_total(args.carry_cap, args.B_total, args.f_full, args.mature_affy, args.U)
        # calculate the parameters for the logistic function
        args.Lp = selection_utils.find_Lp(args.f_full, args.U)

    run_simulation(args)

# ----------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
