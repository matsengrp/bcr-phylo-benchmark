#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''
This module contains classes for simulation and inference for a binary branching process with mutation
in which the tree is collapsed to nodes that count the number of clonal leaves of each type
'''

from __future__ import division, print_function
import scipy, random, pandas as pd, os, time
import itertools
from scipy.stats import poisson
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


class MutationModel():
    '''
    A class for a mutation model, and functions to mutate sequences.
    '''
    def __init__(self, args, mutability_file=None, substitution_file=None, mutation_order=True, allow_re_mutation=True):
        """
        initialized with input files of the S5F format
        @param mutation_order: whether or not to mutate sequences using a context sensitive manner
                               where mutation order matters
        @param allow_re_mutation: allow the same position to mutate multiple times on a single branch
        """
        self.args = args
        self.mutation_order = mutation_order
        self.allow_re_mutation = allow_re_mutation
        if mutability_file is not None and substitution_file is not None:
            self.context_model = {}
            with open(mutability_file, 'r') as f:
                # Eat header:
                f.readline()
                for line in f:
                    motif, score = line.replace('"', '').split()[:2]
                    self.context_model[motif] = float(score)

            # kmer k
            self.k = None
            with open(substitution_file, 'r') as f:
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

    def mutabilities(self, sequence):
        '''Returns the mutability of a sequence at each site, along with nucleotide biases.'''
        # Pad with Ns to allow averaged edge effects:
        sequence = 'N'*(self.k//2) + sequence + 'N'*(self.k//2)
        # Mutabilities of each nucleotide:
        return [self.mutability(sequence[(i-self.k//2):(i+self.k//2+1)]) for i in range(self.k//2, len(sequence) - self.k//2)]

    def mutate(self, sequence, lambda0=1, return_n_mutations=False, debug=False):
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
        n_mutations = scipy.random.poisson(lambda_sequence)
        if not self.allow_re_mutation:
            trials = 20
            for trial in range(1, trials+1):
                n_mutations = scipy.random.poisson(lambda_sequence)
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
    def make_one_mutant(self, sequence, Nmuts, lambda0=0.1):
        '''
        Make a single mutant with a hamming distance, in amino acid space, of Nmuts away from the starting point.
        '''
        trial = 100  # Allow 100 trials before quitting
        while trial > 0:
            mut_seq = sequence[:]
            aa = translate(sequence)
            aa_mut = translate(mut_seq)
            dist = hamming_distance(aa, aa_mut)
            while dist < Nmuts:
                mut_seq = self.mutate(mut_seq, lambda0=lambda0)
                aa_mut = translate(mut_seq)
                dist = hamming_distance(aa, aa_mut)
            if dist == Nmuts and '*' not in aa_mut:  # Stop codon cannot be part of the return
                return aa_mut
            else:
                trial -= 1

        raise RuntimeError('100 consecutive attempts for creating a target sequence failed.')

    # ----------------------------------------------------------------------------------------
    def set_observation_frequencies_and_names(self, tree, obs_times, n_to_downsample, current_time, final_leaves):
        tree.get_tree_root().name = 'naive'  # doesn't seem to get written properly

        potential_names, used_names = None, None

        if obs_times is not None and len(obs_times) > 1:  # observe all intermediate sampled nodes
            for node in [l for l in tree.iter_descendants() if l.sampled]:
                node.frequency = 1
                uid, potential_names, used_names = selection_utils.choose_new_uid(potential_names, used_names, initial_length=3)
                node.name = 'int-' + uid

        if n_to_downsample is not None and len(final_leaves) > n_to_downsample[-1]:  # if we were asked to downsample, and if there's enough leaves to do so
            final_leaves = random.sample(final_leaves, n_to_downsample[-1])

        for leaf in final_leaves:
            leaf.frequency = 1
            uid, potential_names, used_names = selection_utils.choose_new_uid(potential_names, used_names, initial_length=3)
            leaf.name = 'leaf-' + uid

        if self.args.observe_common_ancestors:
            observed_nodes = [n for n in tree.iter_descendants() if n.frequency == 1]
            for node_1, node_2 in itertools.combinations(observed_nodes, 2):
                mrca = node_1.get_common_ancestor(node_2)
                # print('    %s, %s:  %s' % (node_1.name, node_2.name, mrca.name))
                mrca.frequency = 1
                uid, potential_names, used_names = selection_utils.choose_new_uid(potential_names, used_names, initial_length=3)
                mrca.name = 'mrca-' + uid

    # ----------------------------------------------------------------------------------------
    def simulate(self, sequence, pair_bounds=None, lambda_=0.9, lambda0=[1],
                 n_final_seqs=None, obs_times=None, n_to_downsample=None, verbose=False, selection_params=None):
        '''
        Simulate a poisson branching process with mutation introduced
        by the chosen mutation model e.g. motif or uniform.
        Can either simulate under a neutral model without selection,
        or using an affinity muturation inspired model for selection.
        '''
        progeny = poisson(lambda_)  # Default progeny distribution
        stop_dist = None  # Default stopping criterion for affinity simulation

        # Planting the tree:
        tree = TreeNode()
        tree.dist = 0
        tree.add_feature('sequence', sequence)
        tree.add_feature('terminated', False)
        tree.add_feature('sampled', False)  # set if it's sampled at an intermediate time point
        tree.add_feature('frequency', 0)  # observation frequency, seems to always be either 1 or 0
        tree.add_feature('time', 0)

        if selection_params is not None:
            hd_generation = list()  # Collect an array of the counts of each hamming distance (to a target) at each time step
            stop_dist, mature_affy, naive_affy, target_distance, target_count, skip_update, A_total, B_total, Lp, k_exp, outbase = selection_params

            print('    making %d target sequences' % target_count)
            targetAAseqs = [self.make_one_mutant(sequence, target_distance) for i in range(target_count)]

            aa_naive_seq = translate(tree.sequence)
            if '*' in aa_naive_seq:
                raise Exception('stop codon in naive sequence AA translation (this isn\'t necessarily otherwise forbidden, but it\'ll quickly end you in a thicket of infinite loops)')
            assert len(set([len(aa_naive_seq)] + [len(t) for t in targetAAseqs])) == 1  # targets and naive seq are same length
            assert len(set([target_distance] + [hamming_distance(aa_naive_seq, t) for t in targetAAseqs])) == 1  # all targets are the right distance from the naive

            assert target_distance > 0
            def hd2affy(hd):  # affinity is an exponential function of hamming distance
                return(mature_affy + hd**k_exp * (naive_affy - mature_affy) / target_distance**k_exp)

            tree.add_feature('AAseq', str(aa_naive_seq))
            tree.add_feature('Kd', selection_utils.calc_Kd(tree.AAseq, targetAAseqs, hd2affy))
            tree.add_feature('target_distance', target_distance)

        print('    starting %d generations' % max(obs_times))
        if verbose:
            print('       gen   live leaves')
            print('             before/after   ileaf   n children     n mutations          Kd   (%s: updated lambdas)' % selection_utils.color('blue', 'x'))
        current_time = 0
        n_unterminated_leaves = 1
        lambda_min = 10e-10  # small lambdas are causing problems so make a minimum
        hd_distrib = []
        while True:
            if n_unterminated_leaves <= 0:  # if everybody's dead (probably can't actually go less than zero, but not sure)
                break
            if n_final_seqs is not None and n_unterminated_leaves >= n_final_seqs:  # if we've got as many sequences as we were asked for
                break
            if obs_times is not None and current_time >= max(obs_times):  # if we've done as many generations as we were told to
                break
            if stop_dist is not None and current_time > 0 and stop_dist < min(hd_distrib):  # if the leaves have gotten close enough to the target sequences
                break

            # sample any requested intermediate time points (from *last* generation, since haven't yet incremented current_time)
            if obs_times is not None and len(obs_times) > 1 and current_time in obs_times:
                assert len(obs_times) == len(n_to_downsample)
                n_to_sample = n_to_downsample[obs_times.index(current_time)]
                live_nostop_leaves = [l for l in tree.iter_leaves() if not l.terminated and not has_stop(l.sequence)]
                if len(live_nostop_leaves) < n_to_sample:
                    raise RuntimeError('tried to sample %d leaves at intermediate timepoint %d, but tree only has %d live leaves without stops (try a later generation or larger carrying capacity).' % (n_to_sample, current_time, len(live_nostop_leaves)))
                for leaf in random.sample(live_nostop_leaves, n_to_sample):
                    leaf.sampled = True
                    if self.args.kill_sampled_intermediates:
                        leaf.terminated = True
                        n_unterminated_leaves -= 1
                if verbose:
                    print('                  sampled %d (of %d live and stop-free) intermediate leaves (%s) at time %d (but time is about to increment to %d)' % (n_to_sample, len(live_nostop_leaves), 'killing each of them' if self.args.kill_sampled_intermediates else 'leaving them alive', current_time, current_time + 1))

            current_time += 1

            skip_lambda_n = 0  # index keeping track of how many leaves since we last updated all the lambdas
            live_leaves = [l for l in tree.iter_leaves() if not l.terminated]
            random.shuffle(live_leaves)
            if verbose:
                print('      %3d    %3d' % (current_time, len(live_leaves)), end='\n' if len(live_leaves) == 0 else '')  # NOTE these are live leaves *after* the intermediate sampling above
            for leaf in live_leaves:
                if selection_params is not None:
                    lambda_update_dbg_str = ' '
                    if skip_lambda_n == 0:  # time to update the lambdas for every leaf
                        skip_lambda_n = skip_update + 1  # reset it so we don't update again until we've done <skip_update> more leaves (+ 1 is so that if skip_update is 0 we don't skip at all, i.e. skip_update is the number of leaves skipped, *not* the number of leaves *between* updates)
                        tree = selection_utils.update_lambda_values(tree, targetAAseqs, hd2affy, A_total, B_total, Lp)
                        lambda_update_dbg_str = selection_utils.color('blue', 'x')
                    progeny = poisson(max(leaf.lambda_, lambda_min))
                    skip_lambda_n -= 1

                n_children = int(progeny.rvs())
                if current_time == 1 and len(live_leaves) == 1:  # if the first leaf draws zero children, keep trying so we don't have to throw out the whole tree and start over
                    itry = 0
                    while n_children == 0:
                        n_children = int(progeny.rvs())
                        itry += 1
                        if itry > 10 == 0:
                            print('too many tries to get at least one child, giving up on tree')
                            break

                n_unterminated_leaves += n_children - 1  # <-- Getting 1, is equal to staying alive
                if n_children == 0:
                    leaf.terminated = True
                    if len(live_leaves) == 1:
                        print('  terminating only leaf with no children')

                if verbose:
                    n_mutation_list, kd_list = [], []
                for _ in range(n_children):
                    if pair_bounds is not None:  # for paired heavy/light we mutate them separately with their own mutation rate
                        mutated_sequence1 = self.mutate(leaf.sequence[pair_bounds[0][0]:pair_bounds[0][1]], lambda0=lambda0[0])
                        mutated_sequence2 = self.mutate(leaf.sequence[pair_bounds[1][0]:pair_bounds[1][1]], lambda0=lambda0[1])
                        mutated_sequence = mutated_sequence1 + mutated_sequence2
                    else:
                        mutated_sequence, n_muts = self.mutate(leaf.sequence, lambda0=lambda0[0], return_n_mutations=True)
                        if verbose:
                            n_mutation_list.append(n_muts)
                    child = TreeNode()
                    child.dist = sum(x!=y for x,y in zip(mutated_sequence, leaf.sequence))  # NOTE the .dist feature later gets changed to AA distance
                    child.add_feature('sequence', mutated_sequence)
                    if selection_params is not None:
                        child.add_feature('AAseq', str(translate(child.sequence)))
                        child.add_feature('Kd', selection_utils.calc_Kd(child.AAseq, targetAAseqs, hd2affy))
                        child.add_feature('target_distance', min([hamming_distance(child.AAseq, taa) for taa in targetAAseqs]))
                        if verbose:
                            kd_list.append(child.Kd)
                    child.add_feature('frequency', 0)  # observation frequency, seems to always be either 1 or 0
                    child.add_feature('terminated', False)
                    child.add_feature('sampled', False)  # set if it's sampled at an intermediate time point
                    child.add_feature('time', current_time)
                    leaf.add_child(child)
                if verbose:
                    n_mutation_str_list = [('%d' % n) if n > 0 else '-' for n in n_mutation_list]
                    kd_str_list = ['%.0f' % kd for kd in kd_list]
                    pre_leaf_str = '' if live_leaves.index(leaf) == 0 else ('%12s %3d' % ('', len([l for l in tree.iter_leaves() if not l.terminated])))
                    print(('      %s      %4d   %3d  %s          %-14s       %-28s') % (pre_leaf_str, live_leaves.index(leaf), n_children, lambda_update_dbg_str, ' '.join(n_mutation_str_list), ' '.join(kd_str_list)))
            if selection_params is not None:
                hd_distrib = [min([hamming_distance(tn.AAseq, ta) for ta in targetAAseqs]) for tn in tree.iter_leaves() if not tn.terminated]  # list, for each live leaf, of the smallest distance to any target
                n_bins = target_distance * 10 if target_distance > 0 else 10
                hist = scipy.histogram(hd_distrib, bins=list(range(n_bins)))
                hd_generation.append(hist)
                # if verbose and hd_distrib:
                #     print('            total population %-4d    majority distance to target %-3d' % (sum(hist[0]), scipy.argmax(hist[0])))
            # if current_time > 5:
            #     assert False

        # write a histogram of the hamming distances to target at each generation
        if selection_params is not None:
            with open(outbase + '_selection_runstats.p', 'wb') as f:
                pickle.dump(hd_generation, f)

        # check some things
        if obs_times is not None and max(obs_times) != current_time:
            raise RuntimeError('tree terminated at time %d, but we were supposed to sample at time %d' % (current_time, max(obs_times)))
        if n_final_seqs is not None and n_unterminated_leaves < n_final_seqs:
            raise RuntimeError('tree terminated with %d leaves, but --n_final_seqs was set to %d' % (n_unterminated_leaves, n_final_seqs))
        if n_to_downsample is not None and n_unterminated_leaves < n_to_downsample[-1]:
            raise RuntimeError('tree terminated with %d leaves, but --n_to_downsample[-1] was set to %d' % (n_unterminated_leaves, n_to_downsample[-1]))
        if obs_times is not None and len(obs_times) > 1:  # make sure we have the right number of sampled intermediates at each intermediate time point
            for inter_time, n_to_sample in zip(obs_times[:-1], n_to_downsample[:-1]):
                intermediate_sampled_leaves = [l for l in tree.iter_descendants() if l.time == inter_time and l.sampled]  # nodes at this time point that we sampled above
                if len(intermediate_sampled_leaves) < n_to_sample:
                    raise RuntimeError('couldn\'t find the correct number of intermediate sampled leaves at time %d (should have sampled %d, but now we only find %d)' % (inter_time, n_to_sample, len(intermediate_sampled_leaves)))

        stop_leaves = [l for l in tree.iter_leaves() if l.time == current_time and has_stop(l.sequence)]
        non_stop_leaves = [l for l in tree.iter_leaves() if l.time == current_time and not has_stop(l.sequence)]  # non-stop leaves
        if len(stop_leaves) > 0:
            print('    %d / %d leaves at final time point have stop codons' % (len(stop_leaves), len(stop_leaves) + len(non_stop_leaves)))

        self.set_observation_frequencies_and_names(tree, obs_times, n_to_downsample, current_time, non_stop_leaves)

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

        return tree


# ----------------------------------------------------------------------------------------
def run_simulation(args):
    if args.random_seed is not None:
        numpy.random.seed(args.random_seed)
        random.seed(args.random_seed)
    mutation_model = MutationModel(args, args.mutability, args.substitution)  # TODO just pass the whole <args>, and then clean up the remains
    if args.lambda0 is None:
        args.lambda0 = [max([1, int(.01*len(args.sequence))])]
    if args.random_seq is not None:
        from Bio import SeqIO
        records = list(SeqIO.parse(args.random_seq, "fasta"))
        random.shuffle(records)
        args.sequence = str(records[0].seq).upper()
    else:
        args.sequence = args.sequence.upper()
    if args.sequence2 is not None:
        if len(args.lambda0) == 1:  # Use the same mutation rate on both sequences
            args.lambda0 = [args.lambda0[0], args.lambda0[0]]
        elif len(args.lambda0) != 2:
            raise Exception('Only one or two lambda0 can be defined for a two sequence simulation.')

        # Extract the bounds between sequence 1 and 2:
        pair_bounds = ((0, len(args.sequence)), (len(args.sequence), len(args.sequence)+len(args.sequence2)))
        # Merge the two seqeunces to simplify future dealing with the pair:
        args.sequence += args.sequence2.upper()
    else:
        pair_bounds = None
    if args.selection:
        assert(args.B_total >= args.f_full)  # the fully activating fraction on BA must be possible to reach within B_total
        # Find the total amount of A necessary for sustaining the specified carrying capacity
        A_total = selection_utils.find_A_total(args.carry_cap, args.B_total, args.f_full, args.mature_affy, args.U)
        # Calculate the parameters for the logistic function:
        Lp = selection_utils.find_Lp(args.f_full, args.U)
        selection_params = [args.stop_dist, args.mature_affy, args.naive_affy, args.target_distance, args.target_count, args.skip_update, A_total, args.B_total, Lp, args.k_exp, args.outbase]
    else:
        selection_params = None

    trials = 1000
    for trial in range(trials):  # keep trying if we don't get enough leaves, or if there's backmutation
        try:
            tree = mutation_model.simulate(args.sequence,
                                           pair_bounds=pair_bounds,
                                           lambda_=args.lambda_,
                                           lambda0=args.lambda0,
                                           n_to_downsample=args.n_to_downsample,
                                           n_final_seqs=args.n_final_seqs,
                                           obs_times=args.obs_times,
                                           verbose=args.verbose,
                                           selection_params=selection_params)
            if args.selection:
                collapsed_tree = CollapsedTree(tree=tree, name='GCsim selection', collapse_syn=False, allow_repeats=True)
            else:
                collapsed_tree = CollapsedTree(tree=tree, name='GCsim neutral')  # <-- This will fail if backmutations
            tree.ladderize()
            n_observed_seqs = sum(node.frequency > 0 for node in collapsed_tree.tree.traverse())
            if n_observed_seqs < 2:
                raise RuntimeError('collapsed tree contains {} sampled sequences'.format(n_observed_seqs))
            break
        except RuntimeError as e:
            print('      {} {}\n  trying again'.format(selection_utils.color('red', 'error:'), e))
        else:
            raise
    if trial == trials - 1:
        raise RuntimeError('{} attempts exceeded'.format(trials))

    # In the case of a sequence pair print them to separate files:
    if args.sequence2 is not None:
        fh1 = open(args.outbase+'_seq1.fasta', 'w')
        fh2 = open(args.outbase+'_seq2.fasta', 'w')
        fh1.write('>naive\n')
        fh1.write(args.sequence[pair_bounds[0][0]:pair_bounds[0][1]]+'\n')
        fh2.write('>naive\n')
        fh2.write(args.sequence[pair_bounds[1][0]:pair_bounds[1][1]]+'\n')
        for node in tree.iter_descendants():
            if node.frequency != 0:
                fh1.write('>' + node.name + '\n')
                fh1.write(node.sequence[pair_bounds[0][0]:pair_bounds[0][1]]+'\n')
                fh2.write('>' + node.name + '\n')
                fh2.write(node.sequence[pair_bounds[1][0]:pair_bounds[1][1]]+'\n')
    else:
        with open(args.outbase+'.fasta', 'w') as f:
            f.write('>naive\n')
            f.write(args.sequence+'\n')
            for node in tree.iter_descendants():
                if node.frequency != 0:
                    f.write('>' + node.name + '\n')
                    f.write(node.sequence + '\n')

    # Some observable simulation stats to write:
    frequency, distance_from_naive, degree = zip(*[(node.frequency,
                                                    hamming_distance(node.sequence, args.sequence),
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
        colors = {i: next(palette) for i in range(int(len(args.sequence) // 3))}
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

    parser.add_argument('--sequence', type=str, default='GGACCTAGCCTCGTGAAACCTTCTCAGACTCTGTCCCTCACCTGTTCTGTCACTGGCGACTCCATCACCAGTGGTTACTGGAACTGGATCCGGAAATTCCCAGGGAATAAACTTGAGTACATGGGGTACATAAGCTACAGTGGTAGCACTTACTACAATCCATCTCTCAAAAGTCGAATCTCCATCACTCGAGACACATCCAAGAACCAGTACTACCTGCAGTTGAATTCTGTGACTACTGAGGACACAGCCACATATTACTGT',
                        help='Seed naive nucleotide sequence (ignored if --random_seq is set)')
    parser.add_argument('--random_seq', type=str, default=None, help='Path to fasta file containing seed naive sequences. Will draw one of these at random.')
    parser.add_argument('--mutability', type=str, default=file_dir+'/../motifs/Mutability_S5F.csv', help='Path to mutability model file.')
    parser.add_argument('--substitution', type=str, default=file_dir+'/../motifs/Substitution_S5F.csv', help='Path to substitution model file.')
    parser.add_argument('--no_context', action='store_true', help='Disable context dependence, i.e. use a uniform mutability and substitution.')
    parser.add_argument('--selection', action='store_true', help='If set, simulate with selection (otherwise neutral). Requires that you set --obs_times, and therefore that you *not* set --n_final_seqs.')
    parser.add_argument('--n_final_seqs', type=int, default=None, help='If set, simulation stops when we\'ve reached this number of sequences (other stopping criteria: --stop_dist and --obs_times). Because sequences with stop codons are subsequently removed, and because more than on sequence is added per iteration, though we don\'t necessarily output this many. (If --n_to_downsample is also set, then we simulate until we have --n_final_seqs, then downsample to --n_to_downsample).')
    parser.add_argument('--obs_times', type=int, nargs='+', default=None, help='If set, simulation stops when we\'ve reached this many generations. If more than one value is specified, the largest value is the final observation time (and stopping criterion), and earlier values are used as additional, intermediate sampling times (other stopping criteria: --n_final_seqs, --stop_dist)')
    parser.add_argument('--stop_dist', type=int, default=None, help='If set, simulation stops when any simulated sequence is closer than this hamming distance to any of the target sequences (other stopping criteria: --n_final_seqs, --obs_times).')
    parser.add_argument('--lambda', dest='lambda_', type=float, default=.9, help='Poisson branching parameter')
    parser.add_argument('--lambda0', type=float, default=None, nargs='*', help='List of one or two elements with the baseline mutation rates. Space separated input values.\n'
                        'First element belonging to seed sequence one and optionally the next to sequence 2. If only one rate is provided for two sequences, this rate will be used on both.')
    parser.add_argument('--n_to_downsample', type=int, nargs='+', default=None, help='Number of cells sampled during final downsampling step (if only one value is specified), or during each time in --obs_times (if more than one value is specified).')
    parser.add_argument('--kill_sampled_intermediates', action='store_true', help='kill intermediate sequences as they are sampled')
    parser.add_argument('--observe_common_ancestors', action='store_true', help='If set, after deciding which nodes to observe (write to file) according to other options, we then also select the most recent common ancestor for every pair of those nodes.')
    parser.add_argument('--carry_cap', type=int, default=1000, help='The carrying capacity of the simulation with selection. This number affects the fixation time of a new mutation.'
                        'Fixation time is approx. log2(carry_cap), e.g. log2(1000) ~= 10.')
    parser.add_argument('--target_count', type=int, default=10, help='The number of target sequences to generate.')
    parser.add_argument('--target_distance', type=int, default=10, help='Desired distance (number of non-synonymous mutations) between the naive sequence and the target sequences.')
    parser.add_argument('--sequence2', type=str, default=None, help='Second seed naive nucleotide sequence. For simulating heavy/light chain co-evolution.')
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
    parser.add_argument('--outbase', type=str, default='GCsimulator_out', help='Output file base name')
    parser.add_argument('--idlabel', action='store_true', help='Flag for labeling the sequence ids of the nodes in the output tree images, also write associated fasta alignment if True')
    parser.add_argument('--random_seed', type=int, help='for random number generator')

    args = parser.parse_args()
    if args.no_context:
        args.mutability = None
        args.substitution = None
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

    run_simulation(args)

if __name__ == '__main__':
    main()
