#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''
This module contains classes for simulation and inference for a binary branching process with mutation
in which the tree is collapsed to nodes that count the number of clonal leaves of each type
'''

from __future__ import division, print_function
import scipy, random, pandas as pd, os
from itertools import cycle
from scipy.stats import poisson
from ete3 import TreeNode, NodeStyle, TreeStyle, TextFace, CircleFace, PieChartFace, faces, SVG_COLORS
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna
from Bio import AlignIO
from Bio.Phylo.TreeConstruction import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
import matplotlib; matplotlib.use('agg')
try:
    import cPickle as pickle
except:
    import pickle

from utils import hamming_distance, has_stop, translate
import selection_utils

scipy.seterr(all='raise')


class CollapsedTree():
    '''
    Collapsed tree class from GCtree. Collapses an ete3 tree
    into a genotype collapsed tree based on hamming distance between node seqeunces.
    '''
    def __init__(self, params=None, tree=None, collapse_syn=False, allow_repeats=False):
        '''
        For intialization, either params or tree (or both) must be provided
        params: offspring distribution parameters
        tree: ete tree with frequency node feature. If uncollapsed, it will be collapsed.
        '''
        # Collapse synonymous reads:
        if collapse_syn is True:
            tree.dist = 0  # no branch above root
            for node in tree.iter_descendants():
                aa = translate(node.sequence)
                aa_parent = translate(node.up.sequence)
                node.dist = hamming_distance(aa, aa_parent)

        if tree is not None:
            self.tree = tree.copy()
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

    def __str__(self):
        '''Return a string representation for printing.'''
        return 'params = ' + str(self.params)+ '\ntree:\n' + str(self.tree)

    def render(self, outfile, idlabel=False, colormap=None, show_support=False, chain_split=None):
        '''Render to image file, filetype inferred from suffix, svg for color images'''
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
        ts.show_branch_support = show_support
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


class MutationModel():
    '''
    A class for a mutation model, and functions to mutate sequences.
    '''
    def __init__(self, mutability_file=None, substitution_file=None, mutation_order=True, with_replacement=True):
        """
        initialized with input files of the S5F format
        @param mutation_order: whether or not to mutate sequences using a context sensitive manner
                               where mutation order matters
        @param with_replacement: allow the same position to mutate multiple times on a single branch
        """
        self.mutation_order = mutation_order
        self.with_replacement = with_replacement
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
        if self.context_model is None:
            return [(1, dict((n2, 1/3) if n2 is not n else (n2, 0.) for n2 in 'ACGT')) for n in sequence]
        else:
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
        sequence_length = len(sequence)
        if has_stop(sequence):
            raise RuntimeError('Sequence contains stop codon!')

        mutabilities = self.mutabilities(sequence)
        sequence_mutability = sum(mutability[0] for mutability in mutabilities)/sequence_length
        # Poisson rate for this sequence (given its relative mutability):
        lambda_sequence = sequence_mutability*lambda0
        # Number of mutations m:
        trials = 20
        for trial in range(1, trials+1):
            m = scipy.random.poisson(lambda_sequence)
            if m <= sequence_length or self.with_replacement:
                break
            if trial == trials:
                raise RuntimeError('mutations saturating, consider reducing lambda0')

        # Introduce mutations, if causing stop codon, try again, up to 10 times:
        unmutated_positions = range(sequence_length)
        for i in range(m):
            sequence_list = list(sequence)  # make string a list so we can modify it
            # Determine the position to mutate from the mutability matrix:
            mutability_p = scipy.array([mutabilities[pos][0] for pos in unmutated_positions])
            for trial in range(1, trials+1):
                mut_pos = scipy.random.choice(unmutated_positions, p=mutability_p/mutability_p.sum())
                # Now draw the target nucleotide using the substitution matrix
                substitution_p = [mutabilities[mut_pos][1][n] for n in 'ACGT']
                assert 0 <= abs(sum(substitution_p) - 1.) < 1e-10
                chosen_target = scipy.random.choice(4, p=substitution_p)
                original_base = sequence_list[mut_pos]
                sequence_list[mut_pos] = 'ACGT'[chosen_target]
                sequence = ''.join(sequence_list)  # reconstruct our sequence
                # Break the loop if no stop codon:
                if not has_stop(sequence):
                    if self.mutation_order:
                        # If mutation order matters, the mutabilities of the sequence need to be updated:
                        mutabilities = self.mutabilities(sequence)
                    if not self.with_replacement:
                        # Remove this position so we don't mutate it again:
                        unmutated_positions.remove(mut_pos)
                    break
                if trial == trials:
                    raise RuntimeError('stop codon in simulated sequence on '+str(trials)+' consecutive attempts')
                sequence_list[mut_pos] = original_base  # <-- we only get here if we are retrying
        return sequence

    def one_mutant(self, sequence, Nmuts, lambda0=0.1):
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
            if dist == Nmuts:
                return aa_mut
            else:
                trial -= 1
        raise RuntimeError('100 consecutive attempts for creating a target sequence failed.')

    def simulate(self, sequence, pair_bounds=None, lambda_=0.9, lambda0=[1],
                 N=None, T=None, n=None, verbose=False, selection_params=None):
        '''
        Simulate a poisson branching process with mutation introduced
        by the chosen mutation model e.g. motif or uniform.
        Can either simulate under a neutral model without selection,
        or using an affinity muturation inspired model for selection.
        '''
        progeny = poisson(lambda_)  # Default progeny distribution
        stop_dist = None  # Default stopping criterium for affinity simulation
        # Checking the validity of the input parameters:
        if N is not None and T is not None:
            raise ValueError('Only one of N and T can be used. One must be None.')
        if selection_params is not None and T is None:
            raise ValueError('Simulation with selection was chosen. A time, T, must be specified.')
        elif N is None and T is None:
            raise ValueError('Either N or T must be specified.')
        if N is not None and n > N:
            raise ValueError('n ({}) must not larger than N ({})'.format(n, N))

        # Planting the tree:
        tree = TreeNode()
        tree.dist = 0
        tree.add_feature('sequence', sequence)
        tree.add_feature('terminated', False)
        tree.add_feature('frequency', 0)
        tree.add_feature('time', 0)

        if selection_params is not None:
            hd_generation = list()  # Collect an array of the counts of each hamming distance at each time step
            stop_dist, mature_affy, naive_affy, target_dist, skip_update, targetAAseqs, A_total, B_total, Lp, k, outbase = selection_params
            # Assert that the target sequences are comparable to the naive sequence:
            aa = translate(tree.sequence)
            assert(sum([1 for t in targetAAseqs if len(t) != len(aa)]) == 0)  # All targets are same length
            assert(sum([1 for t in targetAAseqs if hamming_distance(aa, t) == target_dist]))  # All target are "target_dist" away from the naive sequence
            # Affinity is an exponential function of hamming distance:
            if target_dist > 0:
                def hd2affy(hd): return(mature_affy + hd**k * (naive_affy - mature_affy) / target_dist**k)
            else:
                def hd2affy(hd): return(mature_affy)
            # We store both the amino acid sequence and the affinity as tree features:
            tree.add_feature('AAseq', str(aa))
            tree.add_feature('Kd', selection_utils.calc_Kd(tree.AAseq, targetAAseqs, hd2affy))
            tree.add_feature('target_dist', min([hamming_distance(tree.AAseq, taa) for taa in targetAAseqs]))

        t = 0  # <-- time
        leaves_unterminated = 1
        # Small lambdas are causing problems so make a minimum:
        lambda_min = 10e-10
        while leaves_unterminated > 0 and (leaves_unterminated < N if N is not None else True) and (t < max(T) if T is not None else True) and (stop_dist >= min(hd_distrib) if stop_dist is not None and t > 0 else True):
            t += 1
            if verbose:
                print('At time:', t)
            skip_lambda_n = 0  # At every new round reset the all the lambdas
            t += 1
            list_of_leaves = list(tree.iter_leaves())
            random.shuffle(list_of_leaves)
            for leaf in list_of_leaves:
                if not leaf.terminated:
                    if selection_params is not None:
                        if skip_lambda_n == 0:
                            skip_lambda_n = skip_update + 1  # Add one so skip_update=0 is no skip
                            tree = selection_utils.lambda_selection(leaf, tree, targetAAseqs, hd2affy, A_total, B_total, Lp)
                        if leaf.lambda_ > lambda_min:
                            progeny = poisson(leaf.lambda_)
                        else:
                            progeny = poisson(lambda_min)
                        skip_lambda_n -= 1
                    n_children = progeny.rvs()
                    leaves_unterminated += n_children - 1  # <-- this kills the parent if we drew a zero
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
                        child.add_feature('time', t)
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

        if selection_params is not None:
            # Keep a histogram of the hamming distances at each generation:
            with open(outbase + 'selection_sim.runstats.p', 'wb') as f:
                pickle.dump(hd_generation, f)

        if leaves_unterminated < N:
            raise RuntimeError('tree terminated with {} leaves, {} desired'.format(leaves_unterminated, N))

        # Each leaf in final generation gets an observation frequency of 1, unless downsampled:
        if T is not None and len(T) > 1:
            # Iterate the intermediate time steps:
            for Ti in sorted(T)[:-1]:
                # Only sample those that have been 'sampled' at intermediate sampling times:
                final_leaves = [leaf for leaf in tree.iter_descendants() if leaf.time == Ti and leaf.sampled]
                if len(final_leaves) < n:
                    raise RuntimeError('tree terminated with {} leaves, less than what desired after downsampling {}'.format(leaves_unterminated, n))
                for leaf in final_leaves:  # No need to down-sample, this was already done in the simulation loop
                    leaf.frequency = 1
        if selection_params and max(T) != t:
            raise RuntimeError('tree terminated with before the requested sample time.')
        # Do the normal sampling of the last time step:
        final_leaves = [leaf for leaf in tree.iter_leaves() if leaf.time == t]
        # By default, downsample to the target simulation size:
        if n is not None and len(final_leaves) >= n:
            for leaf in random.sample(final_leaves, n):
                leaf.frequency = 1
        elif n is None and N is not None:
            for leaf in random.sample(final_leaves, N):
                leaf.frequency = 1
        elif N is None and T is not None:
            for leaf in final_leaves:
                leaf.frequency = 1
        elif n is not None and len(final_leaves) < n:
            raise RuntimeError('tree terminated with {} leaves, less than what desired after downsampling {}'.format(leaves_unterminated, n))
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
    '''
    Simulation subprogram. Can simulate in two modes.
    a) Neutral mode. A Galton–Watson process, with mutation probabilities according to a user defined motif model e.g. S5F.
    b) Selection mode. Using the same mutation process as in a), but in selection mode the poisson progeny distribution's lambda parameter
    is dynamically adjusted accordring to the hamming distance to a list of target sequences. The closer a sequence gets to one of the targets
    the higher fitness and the closer lambda will approach 2, vice versa when the sequence is far away lambda approaches 0.
    '''
    mutation_model = MutationModel(args.mutability, args.substitution)
    if args.lambda0 is None:
        args.lambda0 = [max([1, int(.01*len(args.sequence))])]
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
        # Make a list of target sequences:
        targetAAseqs = [mutation_model.one_mutant(args.sequence, args.target_dist) for i in range(args.target_count)]
        # Find the total amount of A necessary for sustaining the inputted carrying capacity:
        print((args.carry_cap, args.B_total, args.f_full, args.mature_affy))
        A_total = selection_utils.find_A_total(args.carry_cap, args.B_total, args.f_full, args.mature_affy, args.U)
        # Calculate the parameters for the logistic function:
        Lp = selection_utils.find_Lp(args.f_full, args.U)
        selection_params = [args.stop_dist, args.mature_affy, args.naive_affy, args.target_dist, args.skip_update, targetAAseqs, A_total, args.B_total, Lp, args.k, args.outbase]
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
                                           n=args.n,
                                           N=args.N,
                                           T=args.T,
                                           verbose=args.verbose,
                                           selection_params=selection_params)
            if args.selection:
                collapsed_tree = CollapsedTree(tree=tree, collapse_syn=False, allow_repeats=True)
            else:
                collapsed_tree = CollapsedTree(tree=tree)  # <-- this will fail if backmutations
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
        fh1 = open(args.outbase+'_sim_seq1.fasta', 'w')
        fh2 = open(args.outbase+'_sim_seq2.fasta', 'w')
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
        with open(args.outbase+'_sim.fasta', 'w') as f:
            f.write('>naive\n')
            f.write(args.sequence+'\n')
            for leaf in tree.iter_leaves():
                if leaf.frequency != 0:
                    f.write('>' + leaf.name + '\n')
                    f.write(leaf.sequence + '\n')

    # some observable simulation stats to write
    frequency, distance_from_naive, degree = zip(*[(node.frequency,
                                                    hamming_distance(node.sequence, args.sequence),
                                                    sum(hamming_distance(node.sequence, node2.sequence) == 1 for node2 in collapsed_tree.tree.traverse() if node2.frequency and node2 is not node))
                                                   for node in collapsed_tree.tree.traverse() if node.frequency])
    stats = pd.DataFrame({'genotype abundance':frequency,
                          'Hamming distance to root genotype':distance_from_naive,
                          'Hamming neighbor genotypes':degree})
    stats.to_csv(args.outbase+'_sim_stats.tsv', sep='\t', index=False)

    print('{} simulated observed sequences'.format(sum(leaf.frequency for leaf in collapsed_tree.tree.traverse())))

    # render the full lineage tree
    ts = TreeStyle()
    ts.rotation = 90
    ts.show_leaf_name = False
    ts.show_scale = False

    colors = {}
    palette = SVG_COLORS
    palette -= set(['black', 'white', 'gray'])
    palette = cycle(list(palette))  # <-- circular iterator

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

    tree.render(args.outbase+'_sim_lineage_tree.svg', tree_style=ts)

    # render collapsed tree
    # create an id-wise colormap
    # NOTE: node.name can be a set
    if args.plotAA and args.selection:
        colormap = {node.name:colors[node.AAseq] for node in collapsed_tree.tree.traverse()}
    else:
        colormap = {node.name:colors[node.sequence] for node in collapsed_tree.tree.traverse()}
    collapsed_tree.write(args.outbase+'_sim_collapsed_tree.p')
    collapsed_tree.render(args.outbase+'_sim_collapsed_tree.svg',
                          idlabel=args.idlabel,
                          colormap=colormap)
    # print colormap to file
    with open(args.outbase+'_sim_collapsed_tree.colormap.tsv', 'w') as f:
        for name, color in colormap.items():
            f.write((name if isinstance(name, str) else ','.join(name)) + '\t' + color + '\n')


    if args.selection:
        # Define a list a suitable colors that are easy to distinguish:
        palette = ['crimson', 'purple', 'hotpink', 'limegreen', 'darkorange', 'darkkhaki', 'brown', 'lightsalmon', 'darkgreen', 'darkseagreen', 'darkslateblue', 'teal', 'olive', 'wheat', 'magenta', 'lightsteelblue', 'plum', 'gold']
        palette = cycle(list(palette)) # <-- circular iterator
        colors = {i: next(palette) for i in range(int(len(args.sequence) // 3))}
        # The minimum distance to the target is colored:
        colormap = {node.name:colors[node.target_dist] for node in collapsed_tree.tree.traverse()}
        collapsed_tree.write( args.outbase+'_sim_collapsed_runstat_color_tree.p')
        collapsed_tree.render(args.outbase+'_sim_collapsed_runstat_color_tree.svg',
                              idlabel=args.idlabel,
                              colormap=colormap)
        # Write a file with the selection run stats. These are also plotted:
        with open(args.outbase + 'selection_sim_runstats.p', 'rb') as fh:
            runstats = pickle.load(fh)
            selection_utils.plot_runstats(runstats, args.outbase, colors)


def main():
    import argparse
    file_path = os.path.realpath(__file__)
    file_dir = os.path.dirname(file_path)

    parser = argparse.ArgumentParser(description='Germinal center simulation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sequence', type=str, default='GGACCTAGCCTCGTGAAACCTTCTCAGACTCTGTCCCTCACCTGTTCTGTCACTGGCGACTCCATCACCAGTGGTTACTGGAACTGGATCCGGAAATTCCCAGGGAATAAACTTGAGTACATGGGGTACATAAGCTACAGTGGTAGCACTTACTACAATCCATCTCTCAAAAGTCGAATCTCCATCACTCGAGACACATCCAAGAACCAGTACTACCTGCAGTTGAATTCTGTGACTACTGAGGACACAGCCACATATTACTGT',
                        help='Seed naive nucleotide sequence')
    parser.add_argument('--mutability', type=str, default=file_dir+'../S5F/Mutability.csv', help='Path to mutability model file')
    parser.add_argument('--substitution', type=str, default=file_dir+'../S5F/Substitution.csv', help='Path to substitution model file')
    parser.add_argument('--sequence2', type=str, default=None, help='Second seed naive nucleotide sequence. For simulating heavy/light chain co-evolution.')
    parser.add_argument('--lambda', dest='lambda_', type=float, default=.9, help='Poisson branching parameter')
    parser.add_argument('--lambda0', type=float, default=None, nargs='*', help='List of one or two elements with the baseline mutation rates. Space separated input values.'
                        'First element belonging to seed sequence one and optionally the next to sequence 2. If only one rate is provided for two sequences,'
                        'this rate will be used on both.')
    parser.add_argument('--n', type=int, default=None, help='Cells downsampled')
    parser.add_argument('--N', type=int, default=None, help='Target simulation size')
    parser.add_argument('--T', type=int, nargs='+', default=None, help='Observation time, if None we run until termination and take all leaves')
    parser.add_argument('--selection', action='store_true', default=False, help='Simulation with selection? true/false. When doing simulation with selection an observation time cut must be set.')
    parser.add_argument('--stop_dist', type=int, default=None, help='Stop when this distance has been reached in the selection model.')
    parser.add_argument('--carry_cap', type=int, default=1000, help='The carrying capacity of the simulation with selection. This number affects the fixation time of a new mutation.'
                        'Fixation time is approx. log2(carry_cap), e.g. log2(1000) ~= 10.')
    parser.add_argument('--target_count', type=int, default=10, help='The number of targets to generate.')
    parser.add_argument('--target_dist', type=int, default=10, help='The number of non-synonymous mutations the target should be away from the naive.')
    parser.add_argument('--naive_affy', type=float, default=100, help='Affinity of the naive sequence in nano molar.')
    parser.add_argument('--mature_affy', type=float, default=1, help='Affinity of the mature sequences in nano molar.')
    parser.add_argument('--skip_update', type=int, default=100, help='When iterating through the leafs the B:A fraction is recalculated every time.'
                        'It is possible though to update less often and get the same approximate results. This parameter sets the number of iterations to skip,'
                        'before updating the B:A results. skip_update < carry_cap/10 recommended.')
    parser.add_argument('--B_total', type=float, default=1, help='Total number of BCRs per B cell normalized to 10e4. So 1 equals 10e4, 100 equals 10e6 etc.'
                        'It is recommended to keep this as the default.')
    parser.add_argument('--U', type=float, default=5, help='Controls the fraction of BCRs binding antigen necessary to only sustain the life of the B cell'
                        'It is recommended to keep this as the default.')
    parser.add_argument('--f_full', type=float, default=1, help='The fraction of antigen bound BCRs on a B cell that is needed to elicit close to maximum reponse.'
                        'Cannot be smaller than B_total. It is recommended to keep this as the default.')
    parser.add_argument('--k', type=float, default=2, help='The exponent in the function to map hamming distance to affinity.'
                        'It is recommended to keep this as the default.')
    parser.add_argument('--plotAA', action='store_true', default=False, help='Plot trees with collapsing and coloring on amino acid level.')
    parser.add_argument('--verbose', action='store_true', default=False, help='Print progress during simulation. Mostly useful for simulation with selection since this can take a while.')
    parser.add_argument('--outbase', type=str, default='GCsimulator_out', help='Output file base name')
    parser.add_argument('--idlabel', action='store_true', help='Flag for labeling the sequence ids of the nodes in the output tree images, also write associated fasta alignment if True')

    args = parser.parse_args()
    simulate(args)

if __name__ == '__main__':
    main()