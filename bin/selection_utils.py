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

from GCutils import hamming_distance

# ----------------------------------------------------------------------------------------
def calc_kd(seqAA, targetAAseqs, kd_min, kd_max, k_exp, target_distance, fuzz_fraction=None):
    ''' Find the closest target sequence, and apply the transformation function between hamming distance and kd. '''
    assert kd_min < kd_max

    if '*' in seqAA:  # nonsense sequences have zero affinity/infinite kd
        return float('inf')

    distance = min([hamming_distance(seqAA, t) for t in targetAAseqs])  # aa hamming distance to nearest target sequence
    kd = kd_min + (kd_max - kd_min) * (distance / float(target_distance))**k_exp  # apply transformation from distance to kd
    if fuzz_fraction is not None:  # maybe add some fuzz
        assert fuzz_fraction > 0. and fuzz_fraction < 1.
        kd += numpy.random.normal(scale=fuzz_fraction * (kd_max - kd_min))

    return kd

# ----------------------------------------------------------------------------------------
def update_lambda_values(tree, live_leaves, targetAAseqs, A_total, B_total, Lp):
    ''' update the lambda_ feature (parameter for the poisson progeny distribution) for each live leaf in <tree> '''

    def calc_BnA(Kd_n, A, B_total):
        '''
        This calculated the fraction B:A (B bound to A), at equilibrium also referred to as "binding time",
        of all the different Bs in the population given the number of free As in solution.
        '''
        BnA = B_total/(1+Kd_n/A)
        return(BnA)

    def return_objective_A(Kd_n, A_total, B_total):
        '''
        The objective function that solves the set of differential equations setup to find the number of free As,
        at equilibrium, given a number of Bs with some affinity listed in Kd_n.
        '''
        return lambda A: (A_total - (A + scipy.sum(B_total/(1+Kd_n/A))))**2

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
        return(BnA)

    def trans_BA(BA, Lp):
        '''Transform the fraction B:A (B bound to A) to a poisson lambda between 0 and 2.'''
        # We keep alpha to enable the possibility that there is a minimum lambda_:
        alpha, beta, Q = Lp
        lambda_ = alpha + (2 - alpha) / (1 + Q*scipy.exp(-beta*BA))
        return(lambda_)

    # Update the list of affinities for all the live leaves:
    Kd_n = scipy.array([l.Kd for l in live_leaves])
    BnA = calc_binding_time(Kd_n, A_total, B_total)
    lambdas = trans_BA(BnA, Lp)
    for lambda_, leaf in zip(lambdas, live_leaves):
        leaf.lambda_ = lambda_
    return(tree)

# ----------------------------------------------------------------------------------------
def find_A_total(carry_cap, B_total, f_full, mature_kd, U):
    def A_total_fun(A, B_total, Kd_n): return(A + scipy.sum(B_total/(1+Kd_n/A)))

    def C_A(A, A_total, f_full, U): return(U * (A_total - A) / f_full)

    def A_obj(carry_cap, B_total, f_full, Kd_n, U):
        def obj(A): return((carry_cap - C_A(A, A_total_fun(A, B_total, Kd_n), f_full, U))**2)
        return(obj)

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
    return(A_total)


# ----------------------------------------------------------------------------------------
def find_Lp(f_full, U):
    assert(U > 1)
    def T_BA(BA, p):
        # We keep alpha to enable the possibility
        # that there is a minimum lambda_
        alpha, beta, Q = p
        lambda_ = alpha + (2 - alpha) / (1 + Q*scipy.exp(-beta*BA))
        return(lambda_)

    def solve_T_BA(p, f_full, U):
        epsilon = 1/1000
        C1 = (T_BA(0, p) - 0)**2
        C2 = (T_BA(f_full/U, p) - 1)**2
        C3 = (T_BA(1*f_full, p) - (2 - 2*epsilon))**2
        return(C1, C2, C3)

    def solve_T_BA_low_epsilon(p, f_full, U):
        epsilon = 1/1000
        C1 = (T_BA(0, p) - 0)**2
        C2 = (T_BA(f_full/U, p) - 1)**2
        C3 = (T_BA(1*f_full, p) - (2 - 2*epsilon))**2 * ((2 - T_BA(1*f_full, p)) < 2*epsilon)
        return(C1, C2, C3)

    # FloatingPointError errors are not affecting results so ignore them:
    old_settings = scipy.seterr(all='ignore')  # Keep old settings
    scipy.seterr(over='ignore')
    try:
        def obj_T_A(p): return(solve_T_BA(p, f_full, U))
        p = fsolve(obj_T_A, (0, 10e-5, 1), xtol=1e-20, maxfev=1000)
        assert(sum(solve_T_BA(p, f_full, U)) < f_full * 1/1000)
    except:
        print('The U parameter is large and therefore the epsilon parameter has to be adjusted to find a valid solution.')
        def obj_T_A(p): return(solve_T_BA_low_epsilon(p, f_full, U))
        p = fsolve(obj_T_A, (0, 10e-5, 1), xtol=1e-20, maxfev=1000)
        assert(sum(solve_T_BA(p, f_full, U)) < f_full * 1/1000)
    scipy.seterr(**old_settings)  # Reset to default
    return(p)


# ----------------------------------------------------------------------------------------
def plot_runstats(hdist_hists, outbase, colors):
    def make_bounds(hdist_hists):  # hdist_hists: list (over generations) of scipy.hists of min distance to [any] target over leaves
        # scipy.hist is two arrays: [0] is bin counts, [1] is bin x values (not sure if low, high, or centers)
        all_counts = None  # sum over generations of number of leaves in each bin (i.e. at each min distance to target sequence)
        for hist in hdist_hists:
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

    pop_size = scipy.array([sum(r[0]) for r in hdist_hists])  # total population size
    bounds = make_bounds(hdist_hists)  # bin indices of the min and max hamming distances to plot

    fig = plt.figure()
    ax = plt.subplot(111)
    t = scipy.array(list(range(len(pop_size))))  # The x-axis are generations
    ax.plot(t, pop_size, lw=2, label='all cells')  # Total population size is plotted
    # Then plot the counts for each hamming distance as a function on generation:
    for ibin in list(range(*bounds)):
        color = colors[ibin]
        ax.plot(t, scipy.array([r[0][ibin] for r in hdist_hists]), lw=2, color=color, label='distance {}'.format(ibin))

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
