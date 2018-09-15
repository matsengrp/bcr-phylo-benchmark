#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''
This module contains classes for simulation and inference for a binary branching process with mutation
in which the tree is collapsed to nodes that count the number of clonal leaves of each type
'''

from __future__ import division, print_function
import scipy, random, pandas as pd, os, time
from itertools import cycle
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
    def __init__(self, mutability_file=None, substitution_file=None, mutation_order=True, allow_re_mutation=True):
        """
        initialized with input files of the S5F format
        @param mutation_order: whether or not to mutate sequences using a context sensitive manner
                               where mutation order matters
        @param allow_re_mutation: allow the same position to mutate multiple times on a single branch
        """
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

    def mutate(self, sequence, lambda0=1):
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
        lambda_sequence = sequence_mutability*lambda0  # Poisson rate for this sequence (given its relative mutability):

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
        for i in range(n_mutations):
            sequence_list = list(sequence)  # Make mutable

            # Determine the position to mutate from the mutability matrix:
            mutability_p = None
            if self.context_model is not None:
                mutability_p = scipy.array([mutabilities[pos][0] for pos in unmutated_positions])
                mutability_p /= mutability_p.sum()
            mut_pos = scipy.random.choice(unmutated_positions, p=mutability_p)

            # Now draw the target nucleotide using the substitution matrix
            nucs = [n for n in 'ACGT' if n != sequence_list[mut_pos]]
            substitution_p = None
            if self.context_model is not None:
                substitution_p = [mutabilities[mut_pos][1][n] for n in nucs]
                assert 0 <= abs(sum(substitution_p) - 1.) < 1e-10
            sequence_list[mut_pos] = scipy.random.choice(nucs, p=substitution_p)
            sequence = ''.join(sequence_list)  # Reconstruct our string
            if self.context_model is not None and self.mutation_order:  # If mutation order matters, the mutabilities of the sequence need to be updated
                mutabilities = self.mutabilities(sequence)
            if not self.allow_re_mutation:  # Remove this position so we don't mutate it again
                unmutated_positions.remove(mut_pos)

        return sequence

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

    def simulate(self, sequence, pair_bounds=None, lambda_=0.9, lambda0=[1],
                 n_final_seqs=None, obs_times=None, n_to_downsample=None, verbose=False, selection_params=None):
        '''
        Simulate a poisson branching process with mutation introduced
        by the chosen mutation model e.g. motif or uniform.
        Can either simulate under a neutral model without selection,
        or using an affinity muturation inspired model for selection.
        '''
        progeny = poisson(lambda_)  # Default progeny distribution
        stop_dist = None  # Default stopping criterium for affinity simulation
        # Checking the validity of the input parameters:
        if n_final_seqs is not None and obs_times is not None:
            raise ValueError('Only one of N and obs_times can be used. One must be None.')
        if selection_params is not None and obs_times is None:
            raise ValueError('Simulation with selection was chosen. A time, obs_times, must be specified.')
        elif n_final_seqs is None and obs_times is None:
            raise ValueError('Either n_final_seqs or obs_times must be specified.')
        if n_final_seqs is not None and n_to_downsample is not None and n_to_downsample[-1] > n_final_seqs:
            raise ValueError('n ({}) must not larger than n_final_seqs ({})'.format(n_to_downsample[-1], n_final_seqs))
        elif n_final_seqs is not None and n_to_downsample is not None and len(n_to_downsample) != 1:
            raise ValueError('n_to_downsample ({}) must a single value when specifying n_final_seqs'.format(n_to_downsample))
        if obs_times is not None and len(obs_times) > 1 and (n_to_downsample is None or (len(n_to_downsample) != 1 and len(n_to_downsample) != len(obs_times))):
            raise ValueError('n must be specified when using intermediate sampling:', n_to_downsample)
        elif obs_times is not None and len(obs_times) > 1 and len(n_to_downsample) == 1:
            n_to_downsample = [n_to_downsample[-1]] * len(obs_times)

        # Planting the tree:
        tree = TreeNode()
        tree.dist = 0
        tree.add_feature('sequence', sequence)
        tree.add_feature('terminated', False)
        tree.add_feature('sampled', False)
        tree.add_feature('frequency', 0)
        tree.add_feature('time', 0)

        if selection_params is not None:
            hd_generation = list()  # Collect an array of the counts of each hamming distance at each time step
            stop_dist, mature_affy, naive_affy, target_dist, target_count, skip_update, A_total, B_total, Lp, k, outbase = selection_params
            # Make a list of target sequences:
            targetAAseqs = [self.make_one_mutant(sequence, target_dist) for i in range(target_count)]
            # Assert that the target sequences are comparable to the naive sequence:
            aa = translate(tree.sequence)
            assert(sum([1 for t in targetAAseqs if len(t) != len(aa)]) == 0)  # All targets are same length
            assert(sum([1 for t in targetAAseqs if hamming_distance(aa, t) == target_dist]))  # All target are "target_dist" away from the naive sequence
            # Affinity is an exponential function of hamming distance:
            assert(target_dist > 0)
            def hd2affy(hd): return(mature_affy + hd**k * (naive_affy - mature_affy) / target_dist**k)
            # We store both the amino acid sequence and the affinity as tree features:
            tree.add_feature('AAseq', str(aa))
            tree.add_feature('Kd', selection_utils.calc_Kd(tree.AAseq, targetAAseqs, hd2affy))
            tree.add_feature('target_dist', min([hamming_distance(tree.AAseq, taa) for taa in targetAAseqs]))

        t_start = 0  # <-- Time at start
        leaves_unterminated = 1
        # Small lambdas are causing problems so make a minimum:
        lambda_min = 10e-10
        hd_distrib = []
        if not verbose:
            print('generation (out of %d):' % max(obs_times))
        while leaves_unterminated > 0 and (leaves_unterminated < n_final_seqs if n_final_seqs is not None else True) and (t_start < max(obs_times) if obs_times is not None else True) and (stop_dist >= min(hd_distrib) if stop_dist is not None and t_start > 0 else True):
            if verbose:
                print('At time:', t_start)
            else:
                print(' %d' % t_start, end='')
                sys.stdout.flush()
            t_start += 1
            # Sample intermediate time point:
            if obs_times is not None and len(obs_times) > 1 and (t_start-1) in obs_times:
                si = obs_times.index(t_start-1)
                live_nostop_leaves = [l for l in tree.iter_leaves() if not l.terminated and not has_stop(l.sequence)]
                random.shuffle(live_nostop_leaves)
                if len(live_nostop_leaves) < n_to_downsample[si]:
                    raise RuntimeError('tree with {} leaves, less than what desired for intermediate sampling {}. Try later generation or increasing the carrying capacity.'.format(leaves_unterminated, n_to_downsample))
                # Make the sample and kill the cells sampled:
                for leaf in live_nostop_leaves[:n_to_downsample[si]]:
                    leaves_unterminated -= 1
                    leaf.sampled = True
                    leaf.terminated = True
                if verbose:
                    print('Made an intermediate sample at time:', t_start-1)
            live_leaves = [l for l in tree.iter_leaves() if not l.terminated]
            random.shuffle(live_leaves)
            skip_lambda_n = 0  # At every new round reset the all the lambdas
            # Draw progeny for each leaf:
            for leaf in live_leaves:
                if selection_params is not None:
                    if skip_lambda_n == 0:
                        skip_lambda_n = skip_update + 1  # Add one so skip_update=0 is no skip
                        tree = selection_utils.lambda_selection(tree, targetAAseqs, hd2affy, A_total, B_total, Lp)
                    if leaf.lambda_ > lambda_min:
                        progeny = poisson(leaf.lambda_)
                    else:
                        progeny = poisson(lambda_min)
                    skip_lambda_n -= 1
                n_children = progeny.rvs()
                leaves_unterminated += n_children - 1  # <-- Getting 1, is equal to staying alive
                if not n_children:
                    leaf.terminated = True
                for child_count in range(n_children):
                    # If sequence pair mutate them separately with their own mutation rate:
                    if pair_bounds is not None:
                        mutated_sequence1 = self.mutate(leaf.sequence[pair_bounds[0][0]:pair_bounds[0][1]], lambda0=lambda0[0])
                        mutated_sequence2 = self.mutate(leaf.sequence[pair_bounds[1][0]:pair_bounds[1][1]], lambda0=lambda0[1])
                        mutated_sequence = mutated_sequence1 + mutated_sequence2
                    else:
                        mutated_sequence = self.mutate(leaf.sequence, lambda0=lambda0[0])
                    child = TreeNode()
                    child.dist = sum(x!=y for x,y in zip(mutated_sequence, leaf.sequence))
                    child.add_feature('sequence', mutated_sequence)
                    if selection_params is not None:
                        aa = translate(child.sequence)
                        child.add_feature('AAseq', str(aa))
                        child.add_feature('Kd', selection_utils.calc_Kd(child.AAseq, targetAAseqs, hd2affy))
                        child.add_feature('target_dist', min([hamming_distance(child.AAseq, taa) for taa in targetAAseqs]))
                    child.add_feature('frequency', 0)
                    child.add_feature('terminated', False)
                    child.add_feature('sampled', False)
                    child.add_feature('time', t_start)
                    leaf.add_child(child)
            if selection_params is not None:
                hd_distrib = [min([hamming_distance(tn.AAseq, ta) for ta in targetAAseqs]) for tn in tree.iter_leaves() if not tn.terminated]
                if target_dist > 0:
                    hist = scipy.histogram(hd_distrib, bins=list(range(target_dist*10)))
                else:  # Just make a minimum of 10 bins
                    hist = scipy.histogram(hd_distrib, bins=list(range(10)))
                hd_generation.append(hist)
                if verbose and hd_distrib:
                    print('Total cell population:', sum(hist[0]))
                    print('Majority hamming distance:', scipy.argmax(hist[0]))
                    print('Affinity of latest sampled leaf:', leaf.Kd)
                    print('Progeny distribution lambda for the latest sampled leaf:', leaf.lambda_)

        if not verbose:
            print()
        if leaves_unterminated < n_final_seqs:
            raise RuntimeError('Tree terminated with {} leaves, {} desired'.format(leaves_unterminated, n_final_seqs))

        # Keep a histogram of the hamming distances at each generation:
        if selection_params is not None:
            with open(outbase + '_selection_runstats.p', 'wb') as f:
                pickle.dump(hd_generation, f)

        # Each leaf in final generation gets an observation frequency of 1, unless downsampled:
        if obs_times is not None and len(obs_times) > 1:
            # Iterate the intermediate time steps (excluding the last time):
            for Ti in sorted(obs_times)[:-1]:
                si = obs_times.index(Ti)
                # Only sample those that have been 'sampled' at intermediate sampling times:
                final_leaves = [leaf for leaf in tree.iter_descendants() if leaf.time == Ti and leaf.sampled]
                if len(final_leaves) < n_to_downsample[si]:
                    raise RuntimeError('tree terminated with {} leaves, less than what desired after downsampling {}'.format(leaves_unterminated, n_to_downsample[si]))
                for leaf in final_leaves:  # No need to down-sample, this was already done in the simulation loop
                    leaf.frequency = 1
        if selection_params and max(obs_times) != t_start:
            raise RuntimeError('tree terminated before the requested sample time  obs_times: %s    t_start: %d  ' % (obs_times, t_start))
        # Do the normal sampling of the last time step:
        final_leaves = [leaf for leaf in tree.iter_leaves() if leaf.time == t_start and not has_stop(leaf.sequence)]
        # Report stop codon sequences:
        stop_leaves = [leaf for leaf in tree.iter_leaves() if leaf.time == t_start and has_stop(leaf.sequence)]
        if stop_leaves:
            print('Tree contains {} leaves with stop codons, out of {} total at last time point.'.format(len(stop_leaves), len(final_leaves)))

        if obs_times is not None:
            si = obs_times.index(sorted(obs_times)[-1])
        else:
            si = 0
        # By default, downsample to the target simulation size:
        if n_to_downsample is not None and len(final_leaves) >= n_to_downsample[si]:
            for leaf in random.sample(final_leaves, n_to_downsample[si]):
                leaf.frequency = 1
        elif n_to_downsample is None and n_final_seqs is not None:
            if len(final_leaves) < n_final_seqs:  # Removed nonsense sequences might decrease the number of final leaves to less than n_final_seqs
                n_final_seqs = len(final_leaves)
            for leaf in random.sample(final_leaves, n_final_seqs):
                leaf.frequency = 1
        elif n_final_seqs is None and obs_times is not None:
            for leaf in final_leaves:
                leaf.frequency = 1
        elif n_to_downsample is not None and len(final_leaves) < n_to_downsample[si]:
            raise RuntimeError('tree terminated with {} leaves, less than what desired after downsampling {}'.format(leaves_unterminated, n_to_downsample[si]))
        else:
            raise RuntimeError('Unknown option.')

        # Prune away lineages that are unobserved:
        for node in tree.iter_descendants():
            if sum(node2.frequency for node2 in node.traverse()) == 0:
                node.detach()

        # Remove unobserved unifurcations:
        for node in tree.iter_descendants():
            parent = node.up
            if node.frequency == 0 and len(node.children) == 1:
                node.delete(prevent_nondicotomic=False)
                node.children[0].dist = hamming_distance(node.children[0].sequence, parent.sequence)

        # Assign unique names to each node:
        for i, node in enumerate(tree.traverse(), 1):
            node.name = 'simcell_{}'.format(i)

        # Return the uncollapsed tree:
        return tree


def simulate(args):
    if args.random_seed is not None:
        numpy.random.seed(args.random_seed)
        random.seed(args.random_seed)
    mutation_model = MutationModel(args.mutability, args.substitution)
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
        selection_params = [args.stop_dist, args.mature_affy, args.naive_affy, args.target_dist, args.target_count, args.skip_update, A_total, args.B_total, Lp, args.k, args.outbase]
    else:
        selection_params = None

    trials = 1000
    # This loop makes us resimulate if size too small, or backmutation:
    for trial in range(trials):
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
            uniques = sum(node.frequency > 0 for node in collapsed_tree.tree.traverse())
            if uniques < 2:
                raise RuntimeError('collapsed tree contains {} sampled sequences'.format(uniques))
            break
        except RuntimeError as e:
            print('{}, trying again'.format(e))
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
        for leaf in tree.iter_leaves():
            if leaf.frequency != 0:
                fh1.write('>' + leaf.name + '\n')
                fh1.write(leaf.sequence[pair_bounds[0][0]:pair_bounds[0][1]]+'\n')
                fh2.write('>' + leaf.name + '\n')
                fh2.write(leaf.sequence[pair_bounds[1][0]:pair_bounds[1][1]]+'\n')
    else:
        with open(args.outbase+'.fasta', 'w') as f:
            f.write('>naive\n')
            f.write(args.sequence+'\n')
            for leaf in tree.iter_leaves():
                if leaf.frequency != 0:
                    f.write('>' + leaf.name + '\n')
                    f.write(leaf.sequence + '\n')

    # Some observable simulation stats to write:
    frequency, distance_from_naive, degree = zip(*[(node.frequency,
                                                    hamming_distance(node.sequence, args.sequence),
                                                    sum(hamming_distance(node.sequence, node2.sequence) == 1 for node2 in collapsed_tree.tree.traverse() if node2.frequency and node2 is not node))
                                                   for node in collapsed_tree.tree.traverse() if node.frequency])
    stats = pd.DataFrame({'genotype abundance':frequency,
                          'Hamming distance to root genotype':distance_from_naive,
                          'Hamming neighbor genotypes':degree})
    stats.to_csv(args.outbase+'_stats.tsv', sep='\t', index=False)

    print('{} simulated observed sequences'.format(sum(leaf.frequency for leaf in collapsed_tree.tree.traverse())))

    # Render the full lineage tree:
    ts = TreeStyle()
    ts.rotation = 90
    ts.show_leaf_name = False
    ts.show_scale = False

    colors = {}
    palette = SVG_COLORS
    palette -= set(['black', 'white', 'gray'])
    palette = cycle(list(palette))  # <-- Circular iterator

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
        palette = cycle(list(palette)) # <-- circular iterator
        colors = {i: next(palette) for i in range(int(len(args.sequence) // 3))}
        # The minimum distance to the target is colored:
        colormap = {node.name:colors[node.target_dist] for node in collapsed_tree.tree.traverse()}
        collapsed_tree.write( args.outbase+'_collapsed_runstat_color_tree.p')
        collapsed_tree.render(args.outbase+'_collapsed_runstat_color_tree.svg',
                              idlabel=args.idlabel,
                              colormap=colormap)
        # Write a file with the selection run stats. These are also plotted:
        with open(args.outbase + '_selection_runstats.p', 'rb') as fh:
            runstats = pickle.load(fh)
            selection_utils.plot_runstats(runstats, args.outbase, colors)


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
    '''
    parser = argparse.ArgumentParser(description=help_str,
                                     formatter_class=MultiplyInheritedFormatter)

    parser.add_argument('--sequence', type=str, default='GGACCTAGCCTCGTGAAACCTTCTCAGACTCTGTCCCTCACCTGTTCTGTCACTGGCGACTCCATCACCAGTGGTTACTGGAACTGGATCCGGAAATTCCCAGGGAATAAACTTGAGTACATGGGGTACATAAGCTACAGTGGTAGCACTTACTACAATCCATCTCTCAAAAGTCGAATCTCCATCACTCGAGACACATCCAAGAACCAGTACTACCTGCAGTTGAATTCTGTGACTACTGAGGACACAGCCACATATTACTGT',
                        help='Seed naive nucleotide sequence (ignored if --random_seq is set)')
    parser.add_argument('--random_seq', type=str, default=None, help='Path to fasta file containing seed naive sequences. Will draw one of these at random.')
    parser.add_argument('--mutability', type=str, default=file_dir+'/../motifs/Mutability_S5F.csv', help='Path to mutability model file.')
    parser.add_argument('--substitution', type=str, default=file_dir+'/../motifs/Substitution_S5F.csv', help='Path to substitution model file.')
    parser.add_argument('--no_context', action='store_true', help='Disable context dependence, i.e. use a uniform mutability and substitution.')
    parser.add_argument('--sequence2', type=str, default=None, help='Second seed naive nucleotide sequence. For simulating heavy/light chain co-evolution.')
    parser.add_argument('--lambda', dest='lambda_', type=float, default=.9, help='Poisson branching parameter')
    parser.add_argument('--lambda0', type=float, default=None, nargs='*', help='List of one or two elements with the baseline mutation rates. Space separated input values.\n'
                        'First element belonging to seed sequence one and optionally the next to sequence 2. If only one rate is provided for two sequences, this rate will be used on both.')
    parser.add_argument('--n_to_downsample', type=int, nargs='+', default=None, help='Number of cells kept (not discarded) during final downsampling step')
    parser.add_argument('--n_final_seqs', type=int, default=None, help='Desired number of final sequences (actual number may be less due to removal of nonsense sequences)')
    parser.add_argument('--obs_times', type=int, nargs='+', default=None, help='Observation time, in units of rounds of reproduction. Required for selection. If None, run until the settings of other parameters lead to termination, and take all leaves')
    parser.add_argument('--selection', action='store_true', default=False, help='Simulation with selection? true/false. When doing simulation with selection an observation time cut must be set.')
    parser.add_argument('--stop_dist', type=int, default=None, help='Stop when any simulated sequence is closer than this (hamming distance) to any of the target sequences.')
    parser.add_argument('--carry_cap', type=int, default=1000, help='The carrying capacity of the simulation with selection. This number affects the fixation time of a new mutation. '
                        'Fixation time is approx. log2(carry_cap), e.g. log2(1000) ~= 10.')
    parser.add_argument('--target_count', type=int, default=10, help='The number of target sequences to generate.')
    parser.add_argument('--target_dist', type=int, default=10, help='Desired distance (number of non-synonymous mutations) between the naive sequence and the target sequences.')
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
    parser.add_argument('--k', type=float, default=2, help='The exponent in the function to map hamming distance to affinity. '
                        'It is recommended to keep this as the default.')
    parser.add_argument('--plotAA', action='store_true', default=False, help='Plot trees with collapsing and coloring on amino acid level.')
    parser.add_argument('--verbose', action='store_true', default=False, help='Print progress during simulation. Mostly useful for simulation with selection since this can take a while.')
    parser.add_argument('--outbase', type=str, default='GCsimulator_out', help='Output file base name')
    parser.add_argument('--idlabel', action='store_true', help='Flag for labeling the sequence ids of the nodes in the output tree images, also write associated fasta alignment if True')
    parser.add_argument('--random_seed', type=int, help='for random number generator')

    args = parser.parse_args()
    if args.no_context:
        args.mutability = None
        args.substitution = None
    simulate(args)

if __name__ == '__main__':
    main()
