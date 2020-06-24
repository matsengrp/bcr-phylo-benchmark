#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''
Utility functions for selection simulation.
'''

from __future__ import division, print_function

import scipy
import random
import numpy
import itertools
import string
from scipy.optimize import minimize, fsolve
import matplotlib; matplotlib.use('agg')
from matplotlib import pyplot as plt
from ete3 import TreeNode
import operator
import warnings
import os
import csv
import math
warnings.filterwarnings('ignore', 'The iteration is not making good progress')  # scipy.optimize.fsolve() is throwing this. I think it's telling us that our initial guess isn't very good, but as far as I can tell (without understanding kristian's code here) it ends up at a fine solution in the end, so maybe it's ok to turn off this warning

from GCutils import has_stop_aa, hamming_distance, local_translate

# ----------------------------------------------------------------------------------------
# it might make replace_codon_in_aa_seq() faster to use a table here of precached translations, but translation just isn't taking that much time a.t.m.
all_codons = [''.join(c) for c in itertools.product('ACGT', repeat=3)]
all_amino_acids = set(local_translate(c) for c in all_codons)  # note: includes stop codons (*)

# ----------------------------------------------------------------------------------------
def aa_ascii_code_distance(aa1, aa2):  # super arbitrary, but at least for the moment we just want some arbitrary spread in distances
    return abs(ord(aa2) - ord(aa1))

# ----------------------------------------------------------------------------------------
#  Matrix made by matblas from blosum62.iij
#  * column uses minimum score
#  BLOSUM Clustered Scoring Matrix in 1/2 Bit Units
#  Blocks Database = /data/blocks_5.0/blocks.dat
#  Cluster Percentage: >= 62
#  Entropy =   0.6979, Expected =  -0.5209
blosum_text = """   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  B  Z  X  *
                 A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0 -2 -1  0 -4
                 R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3 -1  0 -1 -4
                 N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3  3  0 -1 -4
                 D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3  4  1 -1 -4
                 C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1 -3 -3 -2 -4
                 Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2  0  3 -1 -4
                 E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4
                 G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3 -1 -2 -1 -4
                 H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3  0  0 -1 -4
                 I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3 -3 -3 -1 -4
                 L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1 -4 -3 -1 -4
                 K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2  0  1 -1 -4
                 M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1 -3 -1 -1 -4
                 F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1 -3 -3 -1 -4
                 P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2 -2 -1 -2 -4
                 S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2  0  0  0 -4
                 T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0 -1 -1  0 -4
                 W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3 -4 -3 -2 -4
                 Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1 -3 -2 -1 -4
                 V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4 -3 -2 -1 -4
                 B -2 -1  3  4 -3  0  1 -1  0 -3 -4  0 -3 -3 -2  0 -1 -4 -3 -3  4  1 -1 -4
                 Z -1  0  0  1 -3  3  4 -2  0 -3 -3  1 -1 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4
                 X  0 -1 -1 -1 -2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -2  0  0 -2 -1 -1 -1 -1 -1 -4
                 * -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4  1"""
def init_blosum():
    blines = []
    for line in blosum_text.split('\n'):
        blines.append(line.strip().split())
    top_headers = blines.pop(0)
    left_headers = []
    for bline in blines:
        left_headers.append(bline.pop(0))
    assert top_headers == left_headers
    for ibl, bline in enumerate(blines):
        assert len(bline) == len(top_headers)
        for ival, val in enumerate(bline):
            top_aa = top_headers[ival]
            left_aa = left_headers[ibl]
            if top_aa not in all_amino_acids or left_aa not in all_amino_acids:  # blosum table has ambiguous codes
                continue
            if top_aa == left_aa:  # add them later as a semi-arbitrary value, we have to ignore the values in the table since they're different for different amino acids
                continue
            blinfo[top_aa][left_aa] = -float(val)
    min_val = min(blinfo[aa1][aa2] for aa1 in blinfo for aa2 in blinfo[aa1] if blinfo[aa1][aa2] is not None)
    for aa in all_amino_acids:
        blinfo[aa][aa] = min_val - 1.  # arbitrary, but ok. The diagonal of the matrix varies from -4 to -11, and this gives a.t.m. about -4

blinfo = {aa1 : {aa2 : None for aa2 in all_amino_acids} for aa1 in all_amino_acids}
init_blosum()
sdists = {  # config info necessary for rescaling the two versions of aa similarity distance
    'ascii' : {
        'scale_min' : 0.,  # you want the mean to be kinda sorta around 1, so the --target_dist ends up being comparable
        'scale_max' : 4.7,
    },
    'blosum' : {
        'scale_min' : 0.,
        # 'scale_max' : 3.95,  # use this value if you go back to taking the exp() (note that now the exp() would need to be added in two places)
        'scale_max' : 1.55,
    },
}
for sdtype, sdinfo in sdists.items():
    if sdtype == 'ascii':
        dfcn = aa_ascii_code_distance
    elif sdtype == 'blosum':
        dfcn = lambda aa1, aa2: blinfo.get(aa1).get(aa2)
    else:
        assert False
    vals = [dfcn(aa1, aa2) for aa1, aa2 in itertools.combinations_with_replacement(all_amino_acids, 2)]
    for bound in ['min', 'max']:
        sdinfo[bound] = __builtins__[bound](vals)

# ----------------------------------------------------------------------------------------
def aa_inverse_similarity(aa1, aa2, sdtype, dont_rescale=False):
    if sdtype == 'ascii':
        return_val = aa_ascii_code_distance(aa1, aa2)
    elif sdtype == 'blosum':
        return_val = blinfo[aa1][aa2]
    else:
        assert False
    if not dont_rescale:  # rescale the differences in ascii codes or blosum values for the aa letters to the range [sdists[sdtype]['scale_min'], sdists[sdtype]['scale_max']] (so the mean is around one, so it's similar to the hamming distance)
        sdfo = sdists[sdtype]
        return_val = sdfo['scale_min'] + (return_val - sdfo['min']) * (sdfo['scale_max'] - sdfo['scale_min']) / float(sdfo['max'] - sdfo['min'])
    return return_val

# ----------------------------------------------------------------------------------------
def plot_sdists():
    for aa1 in all_amino_acids:
        print(aa1)
        print('         blosum          ascii')
        print('       raw  rescaled   raw   rescaled')
        for aa2 in all_amino_acids:
            print('   %s  %5.2f  %5.2f    %5.1f  %5.2f' % (aa2,
                                                           aa_inverse_similarity(aa1, aa2, 'blosum', dont_rescale=True), aa_inverse_similarity(aa1, aa2, 'blosum'),
                                                           aa_inverse_similarity(aa1, aa2, 'ascii', dont_rescale=True), aa_inverse_similarity(aa1, aa2, 'ascii')))
    import plotutils
    for sdtype in ['ascii', 'blosum']:
        print(sdtype)
        all_rescaled_vals = [aa_inverse_similarity(aa1, aa2, sdtype=sdtype) for aa1, aa2 in itertools.combinations_with_replacement(all_amino_acids, 2)]
        print(' mean %.3f  min %.1f  max %.1f' % (numpy.mean(all_rescaled_vals), min(all_rescaled_vals), max(all_rescaled_vals)))
        fig, ax = plotutils.mpl_init()
        ax.hist(all_rescaled_vals, bins=45)
        plotutils.mpl_finish(ax, os.getcwd(), sdtype, xlabel='rescaled %s distance' % sdtype, ylabel='AA pairs')

# ----------------------------------------------------------------------------------------
def target_distance_fcn(args, this_seq, target_seqs):
    if args.metric_for_target_distance == 'aa':
        tdists = [(i, hamming_distance(this_seq.dseq('aa'), t.dseq('aa'))) for i, t in enumerate(target_seqs)]  # this is annoyingly complicated because we want to also return *which* target sequence was the closest one, which we have to do here now (instead of afterward) since it depends on which metric we're using
    elif args.metric_for_target_distance == 'nuc':
        tdists = [(i, hamming_distance(this_seq.dseq('nuc'), t.dseq('nuc'))) for i, t in enumerate(target_seqs)]
    elif 'aa-sim' in args.metric_for_target_distance:
        assert len(args.metric_for_target_distance.split('-')) == 3
        sdtype = args.metric_for_target_distance.split('-')[2]
        tdists = [(i, sum(aa_inverse_similarity(aa1, aa2, sdtype) for aa1, aa2 in zip(this_seq.dseq('aa'), t.dseq('aa')) if aa1 != aa2)) for i, t in enumerate(target_seqs)]
    else:
        raise Exception('unsupported --metric_for_target_distance \'%s\'' % args.metric_for_target_distance)
    itarget, tdist = min(tdists, key=operator.itemgetter(1))
    return itarget, tdist

# ----------------------------------------------------------------------------------------
def calc_kd(node, args):
    if has_stop_aa(node.aa_seq):  # nonsense sequences have zero affinity/infinite kd
        return float('inf')

    assert args.mature_kd < args.naive_kd
    tdist = node.target_distance if args.min_target_distance is None else max(node.target_distance, args.min_target_distance)
    kd = args.mature_kd + (args.naive_kd - args.mature_kd) * (tdist / float(args.target_distance))**args.k_exp  # transformation from distance to kd

    return kd

# ----------------------------------------------------------------------------------------
def update_lambda_values(live_leaves, A_total, B_total, logi_params, selection_strength, lambda_min=10e-10):
    ''' update the lambda_ feature (parameter for the poisson progeny distribution) for each leaf in <live_leaves> '''

    # ----------------------------------------------------------------------------------------
    def calc_BnA(Kd_n, A, B_total):
        '''
        This calculates the fraction B:A (B bound to A), at equilibrium also referred to as "binding time",
        of all the different Bs in the population given the number of free As in solution.
        '''
        BnA = B_total/(1+Kd_n/A)
        return(BnA)

    # ----------------------------------------------------------------------------------------
    def return_objective_A(Kd_n, A_total, B_total):
        '''
        The objective function that solves the set of differential equations setup to find the number of free As,
        at equilibrium, given a number of Bs with some affinity listed in Kd_n.
        '''
        return lambda A: (A_total - (A + scipy.sum(B_total/(1+Kd_n/A))))**2

    # ----------------------------------------------------------------------------------------
    def calc_binding_time(Kd_n, A_total, B_total):
        '''
        Solves the objective function to find the number of free As and then uses this,
        to calculate the fraction B:A (B bound to A) for all the different Bs.
        '''
        obj = return_objective_A(Kd_n, A_total, B_total)
        # Different minimizers have been tested and 'L-BFGS-B' was significant faster than anything else:
        obj_min = minimize(obj, A_total, bounds=[[1e-10, A_total]], method='L-BFGS-B', tol=1e-20)
        BnA = calc_BnA(Kd_n, obj_min.x[0], B_total)
        # Terminate if the precision is not good enough:
        assert(BnA.sum()+obj_min.x[0]-A_total < A_total/100)
        return BnA

    # ----------------------------------------------------------------------------------------
    def trans_BA(BnA):
        '''Transform the fraction B:A (B bound to A) to a poisson lambda between 0 and 2.'''
        # We keep alpha to enable the possibility that there is a minimum lambda_:
        lambda_ = alpha + (2 - alpha) / (1 + Q*scipy.exp(-beta*BnA))
        return [max(lambda_min, l) for l in lambda_]

    # ----------------------------------------------------------------------------------------
    # note that since this scales to the existing (kd-determined) mean lambda, if that lambda is decreasing (for instance if selection strength is very low, then sequences drift away from the target very rapidly) then the rescaled lambdas will also decrease
    def apply_selection_strength_scaling(lambdas):  # if <selection_strength> less than 1, instead of using each cell's Kd-determined lambda value, we draw each cell's lambda from a normal distribution with mean and variance depending on the selection strength, that cell's Kd-determined lambda, and the un-scaled distribution of lambda over cells
        def getmean(lvals):
            return numpy.mean([l for l in lvals if l > lambda_min])  # mean unscaled lambda of functional cells (unscaled means determined solely by each cell's Kd)
        def getvar(lvals):
            functional_lambdas = [l for l in lvals if l > lambda_min]
            return numpy.std(functional_lambdas, ddof=1 if len(functional_lambdas)>1 else 0)  # mean unscaled variance of functional cells
        assert selection_strength >= 0. and selection_strength < 1.
        lmean, lvar = getmean(lambdas), getvar(lambdas)
        imeanvals, ivarvals = [], []
        for il, ilambda in enumerate(lambdas):  # draw each cell's lambda from a normal (<ilambda> is the unscaled lambda for this cell, i.e. determined solely by this cell's Kd)
            if ilambda > lambda_min:  # functional cells
                imeanvals.append(lmean + selection_strength * (ilambda - lmean))  # mean of each cell's normal distribution goes from <lmean> to <ilambda> as <selection_strength> goes from 0 to 1
                ivarvals.append((1. - selection_strength) * lvar)  # this gives a variance equal to <lvar> (which is a somewhat arbitrary choice, but I don't think we care too much to change the overall variance) for <selection_strength> near 0 and 1, but the resulting variance drops to about 0.7 of <lvar> for <selection_strength> near 0.5 # TODO fix this (adding a math.sqrt is a bit better) NOTE this got a lot better after I started treating the infinite-kd cells correclty (now it's like 0.9ish for 0.5
            else:  # cells with (presumably) stop codons
                imeanvals.append(lambda_min)  # this is a little weird and wasteful, but if we did it differently we'd have to keep track of which indices have what
                ivarvals.append(0)
        lambdas = numpy.random.normal(imeanvals, ivarvals)  # this is about twice as fast as doing them individually, although this fcn is only 10-20% of the total simulation time
        # if lmean > 0 and lvar > 0:
        #     print('    %5d   %7.4f  %7.4f --> %7.4f  %7.4f' % (len(lambdas), lmean, lvar, getmean(lambdas) / lmean, getvar(lambdas) / lvar))
        return [max(lambda_min, l) for l in lambdas]

    # ----------------------------------------------------------------------------------------
    alpha, beta, Q = logi_params
    Kd_n = scipy.array([l.Kd for l in live_leaves])
    BnA = calc_binding_time(Kd_n, A_total, B_total)  # get list of binding time values for each cell
    new_lambdas = trans_BA(BnA)  # convert binding time list to list of poisson lambdas for each cell (which determine number of offspring)
    if selection_strength < 1:
        new_lambdas = apply_selection_strength_scaling(new_lambdas)
    for new_lambda, leaf in zip(new_lambdas, live_leaves):  # transfer new lambda values to the actual leaves
        leaf.lambda_ = new_lambda
    return new_lambdas

# ----------------------------------------------------------------------------------------
def find_A_total(carry_cap, B_total, f_full, mature_kd, U):
    # find the total amount of A necessary for sustaining the specified carrying capacity
    def A_total_fun(A, B_total, Kd_n): return(A + scipy.sum(B_total/(1+Kd_n/A)))

    def C_A(A, A_total, f_full, U): return(U * (A_total - A) / f_full)

    def A_obj(carry_cap, B_total, f_full, Kd_n, U):
        def obj(A): return((carry_cap - C_A(A, A_total_fun(A, B_total, Kd_n), f_full, U))**2)
        return obj

    Kd_n = scipy.array([mature_kd] * carry_cap)
    obj = A_obj(carry_cap, B_total, f_full, Kd_n, U)
    # Some funny "zero encountered in true_divide" errors are not affecting results so ignore them:
    old_settings = scipy.seterr(all='ignore')  # Keep old settings
    scipy.seterr(divide='ignore')
    obj_min = minimize(obj, 1e-20, bounds=[[1e-20, carry_cap]], method='L-BFGS-B', tol=1e-20)
    scipy.seterr(**old_settings)  # Reset to default
    A = obj_min.x[0]
    A_total = A_total_fun(A, B_total, Kd_n)
    assert(C_A(A, A_total, f_full, U) > carry_cap * 99/100)
    return A_total


# ----------------------------------------------------------------------------------------
# calculate the parameters for the logistic function, i.e. (alpha, beta, Q)
def find_logistic_params(f_full, U):
    assert(U > 1)
    def T_BA(BA, lparams):
        # We keep alpha to enable the possibility
        # that there is a minimum lambda_
        alpha, beta, Q = lparams
        lambda_ = alpha + (2 - alpha) / (1 + Q*scipy.exp(-beta*BA))
        return(lambda_)

    def solve_T_BA(lparams, f_full, U):
        epsilon = 1/1000
        C1 = (T_BA(0, lparams) - 0)**2
        C2 = (T_BA(f_full/U, lparams) - 1)**2
        C3 = (T_BA(1*f_full, lparams) - (2 - 2*epsilon))**2
        return(C1, C2, C3)

    def solve_T_BA_low_epsilon(lparams, f_full, U):
        epsilon = 1/1000
        C1 = (T_BA(0, lparams) - 0)**2
        C2 = (T_BA(f_full/U, lparams) - 1)**2
        C3 = (T_BA(1*f_full, lparams) - (2 - 2*epsilon))**2 * ((2 - T_BA(1*f_full, lparams)) < 2*epsilon)
        return(C1, C2, C3)

    def run_fsolve(obj_T_A):
        return fsolve(obj_T_A, (0, 10e-5, 1), xtol=1e-20, maxfev=1000)

    # FloatingPointError errors are not affecting results so ignore them:
    old_settings = scipy.seterr(all='ignore')  # Keep old settings
    scipy.seterr(over='ignore')
    def obj_T_A(lparams): return(solve_T_BA(lparams, f_full, U))
    try:
        lparams = run_fsolve(obj_T_A)
    except:
        print('  %s U parameter too large when calculating logistic function parameters, so adjusting epsilon parameter to get a valid solution.' % (color('yellow', 'warning')))
        def obj_T_A(lparams): return(solve_T_BA_low_epsilon(lparams, f_full, U))
        lparams = run_fsolve(obj_T_A)
    assert(sum(solve_T_BA(lparams, f_full, U)) < f_full * 1/1000)
    scipy.seterr(**old_settings)  # Reset to default
    return lparams  # tuple with (alpha, beta, Q)

# ----------------------------------------------------------------------------------------
def plot_runstats(tdist_hists, outbase, colors):
    def make_bounds(tdist_hists):  # tdist_hists: list (over generations) of scipy.hists of min distance to [any] target over leaves
        # scipy.hist is two arrays: [0] is bin counts, [1] is bin x values (not sure if low, high, or centers)
        all_counts = None  # sum over generations of number of leaves in each bin (i.e. at each min distance to target sequence)
        for hist in tdist_hists:
            if all_counts is None:
                all_counts = hist[0].copy()
            else:
                all_counts += hist[0]
        imin, imax = None, None
        for j, count in enumerate(all_counts):
            if imin is None and count > 0:
                imin = j
            elif count > 0:
                imax = j
        return(imin, imax)

    tdist_hists = [h for h in tdist_hists if h is not None]  # the initial list of hist is filled with none values to length max(args.obs_times), but then if we stop because of another criterion some of the nones are still there, so i guess just remove them here for plotting
    pop_size = scipy.array([sum(r[0]) for r in tdist_hists])  # total population size
    bounds = make_bounds(tdist_hists)  # bin indices of the min and max hamming distances to plot
    if None in bounds:
        print('  note: couldn\'t get bounds for runstat hists, so not writing')
        return

    fig = plt.figure()
    ax = plt.subplot(111)
    t = scipy.array(list(range(len(pop_size))))  # The x-axis are generations
    ax.plot(t, pop_size, lw=2, label='all cells')  # Total population size is plotted
    # Then plot the counts for each hamming distance as a function on generation:
    for ibin in list(range(*bounds)):
        color = colors[ibin]
        ax.plot(t, scipy.array([r[0][ibin] for r in tdist_hists]), lw=2, color=color, label='distance {}'.format(ibin))

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)

    # Shrink current axis by 20% to make the legend fit:
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    plt.ylabel('count')
    plt.xlabel('GC generation')
    plt.title('population over time of cells grouped by min. distance to target')
    fig.savefig(outbase + '.selection_sim.runstats.pdf')

# ----------------------------------------------------------------------------------------
# bash color codes
Colors = {}
Colors['head'] = '\033[95m'
Colors['bold'] = '\033[1m'
Colors['purple'] = '\033[95m'
Colors['blue'] = '\033[94m'
Colors['light_blue'] = '\033[1;34m'
Colors['green'] = '\033[92m'
Colors['yellow'] = '\033[93m'
Colors['red'] = '\033[91m'
Colors['reverse_video'] = '\033[7m'
Colors['red_bkg'] = '\033[41m'
Colors['end'] = '\033[0m'

# ----------------------------------------------------------------------------------------
def color(col, seq, width=None, padside='left'):
    if col is None:
        return seq
    return_str = [Colors[col], seq, Colors['end']]
    if width is not None:  # make sure final string prints to correct width
        n_spaces = max(0, width - len(seq))  # if specified <width> is greater than uncolored length of <seq>, pad with spaces so that when the colors show up properly the colored sequences prints with width <width>
        if padside == 'left':
            return_str.insert(0, n_spaces * ' ')
        elif padside == 'right':
            return_str.insert(len(return_str), n_spaces * ' ')
        else:
            assert False
    return ''.join(return_str)

# ----------------------------------------------------------------------------------------
def choose_new_uid(potential_names, used_names, initial_length=1, shuffle=False):  # NOTE duplicates code in partis/python/utils.py
    # NOTE only need to set <initial_length> for the first call -- after that if you're reusing the same <potential_names> and <used_names> there's no need (but it's ok to set it every time, as long as it has the same value)
    # NOTE setting <shuffle> will shuffle every time, i.e. it's designed such that you call with shuffle once before starting
    def get_potential_names(length):
        return [''.join(ab) for ab in itertools.combinations(string.ascii_lowercase, length)]
    if potential_names is None:  # first time through
        potential_names = get_potential_names(initial_length)
        used_names = []
    if len(potential_names) == 0:  # ran out of names
        potential_names = get_potential_names(len(used_names[-1]) + 1)
    if len(potential_names[0]) < initial_length:
        raise Exception('choose_new_uid(): next potential name \'%s\' is shorter than the specified <initial_length> %d (this is probably only possible if you called this several times with different <initial_length> values [which you shouldn\'t do])' % (potential_names[0], initial_length))
    if shuffle:
        random.shuffle(potential_names)
    new_id = potential_names.pop(0)
    used_names.append(new_id)
    return new_id, potential_names, used_names
