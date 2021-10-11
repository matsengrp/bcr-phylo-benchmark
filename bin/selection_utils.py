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

#ADD 
from torchdms import utils 
import torch

# ----------------------------------------------------------------------------------------
# it might make replace_codon_in_aa_seq() faster to use a table here of precached translations, but translation just isn't taking that much time a.t.m.
all_codons = [''.join(c) for c in itertools.product('ACGT', repeat=3)]
all_amino_acids = set(local_translate(c) for c in all_codons)  # note: includes stop codons (*)

# ----------------------------------------------------------------------------------------
#MODIFY Load torchdms model and define the onehot encoding function 
data_path = 'data/dummy_file_name.pkl'
data = utils.from_pickle_file(data_path)
wtseq = data.val.wtseq
alphabet = data.val.alphabet

#Might be able to remove the seq_to_onehot function
def seq_to_onehot(seq, alphabet):
    """ Take a given aa-sequence and return it's onehot encoding.
    Args:
        - seq (string): The amino acid sequence to be converted (should be same 
                        length as WT for now)
        - alphabet (dict): A dictionary of characters representing amino acids.
    
    Returns:
        - encoding (torch.Tensor): A 1-hot encoding of `seq`, appropriate for `torchdms`
    """
    # Get indicies to place 1s for present amino acids.
    alphabet_dict = {letter: idx for idx, letter in enumerate(alphabet)}
    seq_idx = [alphabet_dict[aa] for aa in seq]
    indicies = torch.zeros(len(seq), dtype=torch.int, requires_grad=False)
    for site, _ in enumerate(seq):
        indicies[site] = (site * len(alphabet)) + seq_idx[site]
    
    # Generate encoding.
    encoding = torch.zeros((1, len(seq)*len(alphabet)), requires_grad=False)
    for idx in indicies.data:
        encoding[0, idx] = 1
    
    return encoding

# ----------------------------------------------------------------------------------------
#MODIFY changed the calc_kd function to calculate both kd and BCR expression level using torchdms model 
def calc_kd_and_exp(node, args): 
    #want to check if stop codon before loading - QUESTION: will the torchDMS model automatically calculate that? 
    if has_stop_aa(node.aa_seq):  # nonsense sequences have zero affinity/infinite kd
        return [float, 0]
    variant_encoding = seq_to_onehot(node.aa_seq, alphabet)
    variant_preds = model(variant_encoding[0:1])
    #unsure whether the line below is still needed 
    #assert args.mature_kd < args.naive_kd
    kd = torch.flatten(variant_preds)[0]  # transformation from amino acid sequence to kd, arbitrarily used 0 index for kd, unsure in actual model
    exp = torch.flatten(variant_preds)[1]  # transformation from amino acid sequence to BCR expression, need to add an attribute for this 
    return [kd, exp]

# ----------------------------------------------------------------------------------------
def update_lambda_values(live_leaves, A_total, B_total, logi_params, selection_strength, lambda_min=10e-10):
    ''' update the lambda_ feature (parameter for the poisson progeny distribution) for each leaf in <live_leaves> '''

    # ----------------------------------------------------------------------------------------
    #Since each B cell now has its own BCR expression level, must calculate B_total
    def sum_of_B(B_n):
        B_total = scipy.sum(B_n)
        return B_total
    
    def calc_BnA(Kd_n, A, B_n):
        '''
        This calculates the fraction B:A (B bound to A), at equilibrium also referred to as "binding time",
        of all the different Bs in the population given the number of free As in solution.
        '''
        #changed so B_total is B_n
        #original expression BnA = B_total/(1+Kd_n/A)
        BnA = B_n/(1+Kd_n/A)
        return BnA

    # ----------------------------------------------------------------------------------------
    #Changed to use B_n array instead of B_total 
    def return_objective_A(Kd_n, A_total, B_n):
        '''
        The objective function that solves the set of differential equations setup to find the number of free As,
        at equilibrium, given a number of Bs with some affinity listed in Kd_n.
        '''
        return lambda A: (A_total - (A + scipy.sum(B_n/(1+Kd_n/A))))**2  #is it as simple as replacing B_total with B_n here?

    # ----------------------------------------------------------------------------------------
    def calc_binding_time(Kd_n, A_total, B_n):
        '''
        Solves the objective function to find the number of free As and then uses this,
        to calculate the fraction B:A (B bound to A) for all the different Bs.
        '''
        # B_total = sum_of_B(B_n), just here in case B_total is still needed at some point, but currently only using B_n array
        obj = return_objective_A(Kd_n, A_total, B_n)
        # Different minimizers have been tested and 'L-BFGS-B' was significant faster than anything else:
        obj_min = minimize(obj, A_total, bounds=[[1e-10, A_total]], method='L-BFGS-B', tol=1e-20)
        BnA = calc_BnA(Kd_n, obj_min.x[0], B_n)
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
    #Added line to define B_n similarly to Kd_n 
    B_n = scipy.array([l.B_exp for l in live_leaves])
    BnA = calc_binding_time(Kd_n, A_total, B_n)  # get list of binding time values for each cell
    new_lambdas = trans_BA(BnA)  # convert binding time list to list of poisson lambdas for each cell (which determine number of offspring)
    if selection_strength < 1:
        new_lambdas = apply_selection_strength_scaling(new_lambdas)
    for new_lambda, leaf in zip(new_lambdas, live_leaves):  # transfer new lambda values to the actual leaves
        leaf.lambda_ = new_lambda
    return new_lambdas

# ---------------------------------------------------------------------------------------- 
def find_A_total(carry_cap, B_n, f_full, mature_kd, U):
    # find the total amount of A necessary for sustaining the specified carrying capacity
    def A_total_fun(A, some_fixed_B_total, Kd_n): return(A + scipy.sum(some_fixed_B_total/(1+Kd_n/A)))

    def C_A(A, A_total, f_full, U): return(U * (A_total - A) / f_full)

    def A_obj(carry_cap, some_fixed_B_total, f_full, Kd_n, U):
            def obj(A): return((carry_cap - C_A(A, A_total_fun(A, some_fixed_B_total, Kd_n), f_full, U))**2)
            return obj

    Kd_n = scipy.array([mature_kd] * carry_cap)
    obj = A_obj(carry_cap, some_fixed_B_total, f_full, Kd_n, U)
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
#QUESTION: Should we calculate these or directly feed them in as arguments to test our inference capabilities? 
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
def plot_runstats(scatter_value, scatter_index, metric):
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.scatter(scatter_index, scatter_value)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    # Shrink current axis by 20% to make the legend fit:
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    if metric == 'affinity':
        plt.ylabel('-log(affinity)')
    elif metric == 'expression':
        plt.ylabel('expression')
    else:
        plt.ylabel('antigen_capture')
    plt.xlabel('GC generation')
    plt.title('distribution of BCR value at each generation')
    fig.savefig(metric + '.selection_sim.runstats.pdf')

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
