#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Infer trees from germinal center data, and validate inference method with simulation
'''
from __future__ import print_function
import os
import sys
import subprocess
import sconsutils
from warnings import warn
from SCons.Script import Environment, AddOption

sconsutils

# Set up SCons environment
environ = os.environ.copy()
env = Environment(ENV=environ)

# Add stuff to PATH
env.PrependENVPath('PATH', 'bin')

# Setting up command line arguments/options
AddOption('--inference',
          action='store_true',
          help='Run inference')
inference = GetOption('inference')
AddOption('--simulate',
          action='store_true',
          help='Validation subprogram, instead of inference')
simulate = GetOption('simulate')
AddOption('--exp_data_test',
          action='store_true',
          help='Validation subprogram on experimental data e.g. isotype and paired heavy/light chain sequences.')
exp_data_test = GetOption('exp_data_test')
AddOption('--srun',
          action='store_true',
          help='Should jobs be submitted with srun?')
if GetOption('srun'):
    CommandRunner = env.SRun
else:
    CommandRunner = env.Command

# Tool options:
AddOption('--gctree',
           action='store_true',
           help='GCtree inference')
gctree = GetOption('gctree')

AddOption('--igphyml',
           action='store_true',
           help='Use igphyml inference')
igphyml = GetOption('igphyml')

AddOption('--dnaml',
           action='store_true',
           help='Use dnaml inference')
dnaml = GetOption('dnaml')

AddOption('--dnapars',
           action='store_true',
           help='Use dnapars inference')
dnapars = GetOption('dnapars')

AddOption('--samm_rank',
           action='store_true',
           help='Use SAMM to rank dnapars trees.')
samm_rank = GetOption('samm_rank')

AddOption('--mutability',
          type='string',
          default='motifs/Mutability_S5F.csv',
          metavar='PATH',
          help='path to motifs mutability data')
mutability = GetOption('mutability')

AddOption('--substitution',
          type='string',
          default='motifs/Substitution_S5F.csv',
          metavar='PATH',
          help='path to motifs substitution data')
substitution = GetOption('substitution')

AddOption('--uniform_mut',
          action='store_true',
          help='Mutation and substitution drawn from uniform distribution.')
uniform_mut = GetOption('uniform_mut')

AddOption('--fastml',
           action='store_true',
           help='Use FastML inference')
fastml = GetOption('fastml')

AddOption('--iqtree',
           action='store_true',
           help='Use IQ-TREE inference')
iqtree = GetOption('iqtree')
iqtree_options = False
if iqtree:
    AddOption('--iqtree_option_str',
              type='str',
              action='append',
              default=[],
              help='IQ-TREE run options. Can be specified multiple times')
    iqtree_options = GetOption('iqtree_option_str')
    if len(iqtree_options) == 0:
        raise Exception('At least one --iqtree_option_str must be defined when requesting to run IQ-TREE.')

# Fill a dictionary with the tools:
tool_dict = {name: tool for tool, name in zip([gctree, igphyml, dnaml, dnapars, samm_rank, fastml, iqtree_options], ['gctree', 'igphyml', 'dnaml', 'dnapars', 'samm_rank', 'fastml', 'iqtree'])}

AddOption('--outdir',
          type='string',
          help="directory in which to output results")
outdir = GetOption('outdir')
AddOption('--quick',
           action='store_true',
           help='less thourough dnapars tree search (faster)')
quick = GetOption('quick')
AddOption('--idlabel',
           action='store_true',
           help='label sequence ids on tree, and write associated alignment')
idlabel = GetOption('idlabel')
AddOption('--xvfb',
          action='store_true',
          help='use virtual X, for rendering ETE trees on a remote server')
xarg = 'TMPDIR=/tmp xvfb-run -a ' if GetOption('xvfb') else ''
AddOption('--nobuff',
          action='store_true',
          help='use stdbuf to prevent line buffering on linux')
buffarg = 'stdbuf -oL ' if GetOption('nobuff') else ''

if len([True for v in tool_dict.values() if v]) == 0 and not GetOption('help'):
    raise Exception('must set at least one inference method')

if not simulate and not inference and not exp_data_test and not GetOption('help'):
    raise Exception('Please provide one of the required arguments. Either "--inference", "--simulate" or "--exp_data_test".'
                     'Command line help can then be evoked by "-h" or "--help" and found in the bottom'
                     'of the output under "Local Options".')

# Default naive IDs:
naiveIDexp = 'naive0'
naiveID = 'naive'

if simulate:
    AddOption('--naive',
              type='string',
              default='GGACCTAGCCTCGTGAAACCTTCTCAGACTCTGTCCCT'
                      'CACCTGTTCTGTCACTGGCGACTCCATCACCAGTGGTT'
                      'ACTGGAACTGGATCCGGAAATTCCCAGGGAATAAACTT'
                      'GAGTACATGGGGTACATAAGCTACAGTGGTAGCACTTA'
                      'CTACAATCCATCTCTCAAAAGTCGAATCTCCATCACTC'
                      'GAGACACATCCAAGAACCAGTACTACCTGCAGTTGAAT'
                      'TCTGTGACTACTGAGGACACAGCCACATATTACTGT',
              help='sequence of naive from which to simulate')
    naive = GetOption('naive')

    AddOption('--random_naive',
              type='string',
              default=None,
              help='Fasta file with naive sequences from which to draw a random sequence to use as naive.')
    random_naive = GetOption('random_naive')

    AddOption('--lambda',
              type='float',
              action='append',
              default=[],
              help='Poisson branching parameter for simulation')
    lambda_list = GetOption('lambda')
    if len(lambda_list) == 0:
        lambda_list = [2.]

    AddOption('--lambda0',
              type='float',
              action='append',
              default=[],
              help='baseline mutation rate')
    lambda0_list = GetOption('lambda0')
    if len(lambda0_list) == 0:
        lambda0_list = [.25]

    AddOption('--seed',
              type='int',
              default=None,
              help='Simulation seed for random number generation.')
    seed = GetOption('seed')

    AddOption('--n',
              type='int',
              action='append',
              default=[],
              help='Cells downsampled.')
    n = GetOption('n')

    AddOption('--N',
              type='int',
              default=None,
              help='Simulation size (number of cells observerved).')
    N = GetOption('N')

    AddOption('--T',
              type='int',
              action='append',
              default=[],
              help='Sampling time. Can be defined multiple times to make intermediate sampling.')
    T = GetOption('T')

    AddOption('--nsim',
              type='int',
              default=1,
              help='Number of simulations with each parameter parameter choice.')
    nsim = GetOption('nsim')

    AddOption('--experimental',
              type='string',
              action='append',
              default=[],
              help='Experimental fastas for comparing summary stats.')
    experimental_list = GetOption('experimental')

    AddOption('--naiveIDexp',
              type='string',
              default='naive0',
              help='Id of naive seq in the experimental data fasta file. Must be lowercase. If not correct in file.')
    naiveIDexp = GetOption('naiveIDexp')

    AddOption('--selection',
              action='store_true',
              help='Simulation with affinity selection.')
    selection = GetOption('selection')
    selection = '--selection' if selection else ''
    if selection:
        AddOption('--target_dist',
                  type='int',
                  default=10,
                  help='Distance to selection target.')
        target_dist = GetOption('target_dist')

        AddOption('--target_count',
                  type='int',
                  default=10,
                  help='Number of targets.')
        target_count = GetOption('target_count')

        AddOption('--verbose',
                  action='store_true',
                  help='Verbose printing.')
        verbose = GetOption('verbose')
        verbose = '--verbose' if verbose else '' 

        AddOption('--carry_cap',
                  type='int',
                  default=1000,
                  help='Number of targets.')
        carry_cap = GetOption('carry_cap')

        AddOption('--skip_update',
                  type='int',
                  default=100,
                  help='Skip update step.')
        skip_update = GetOption('skip_update')
        selection_param = (target_dist, target_count, carry_cap, skip_update, verbose)
    else:
        selection_param = None

elif exp_data_test:
    AddOption('--exp_data',
              type='string',
              help='Experimental data in CSV format containing isotype and/or chain pairing information.')
    exp_data = GetOption('exp_data')

    AddOption('--naiveIDexp',
              type='string',
              default='naive',
              help='Id of naive seq in the experimental data CSV file. Must be lowercase. If not correct in file.')
    naiveIDexp = GetOption('naiveIDexp')

elif inference:
    AddOption('--naiveID',
              type='string',
              metavar='seqID',
              default='naive',
              help='Id of naive sequence in input fasta. Must be lowercase. If not correct in file.')
    naiveID = GetOption('naiveID')

    AddOption('--converter',
              type='string',
              default=None,
              help='Converter to convert input fasta format e.g. the Victora lab GC fasta format.')
    converter = GetOption('converter')

    AddOption('--fasta',
              dest='fasta',
              type='string',
              default='sequence_data/150228_Clone_3-8.fasta',
              metavar='PATH',
              help='Path to input fasta.')
    fasta = GetOption('fasta')
    # Hard-code the Tas. data converter:
    if fasta == 'sequence_data/150228_Clone_3-8.fasta':
        converter = 'tas'

# Require the naive ID to be lower case and 10 characters or less, because of downstream software compatability:
if naiveIDexp != naiveIDexp.lower() or naiveID != naiveID.lower() or len(naiveIDexp) > 10 or len(naiveID) > 10:
    raise InputError('Naive id must be lowercase.')


# First call after all arguments have been parsed
# to enable correct command line help.
if simulate and not GetOption('help'):
    if outdir is None:
        raise InputError('Outdir must be specified.')
    SConscript('SConscript.simulation',
               exports='env tool_dict quick idlabel outdir naive random_naive mutability substitution lambda_list lambda0_list seed n N T nsim CommandRunner experimental_list naiveIDexp selection_param xarg buffarg uniform_mut')
if exp_data_test and not GetOption('help'):
    if None in [outdir, exp_data]:
        raise InputError('Both outdir and exp_data must be specified.')
    SConscript('SConscript.exp_data_test',
               exports='env tool_dict quick idlabel outdir exp_data naiveIDexp mutability substitution CommandRunner xarg buffarg')
elif inference and not GetOption('help'):
    if None in [fasta, outdir]:
        raise InputError('input fasta and outdir must be specified')
    colormap = None  # Not available with inference
    SConscript('SConscript.inference', exports='env tool_dict quick idlabel fasta outdir naiveID converter CommandRunner xarg buffarg colormap mutability substitution')
