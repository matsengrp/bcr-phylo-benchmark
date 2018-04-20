
from ete3 import TreeNode, TreeStyle, NodeStyle, SVG_COLORS
import pickle
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/bin')
from GCutils import CollapsedTree



def translate(seq):
    return str(Seq(seq[:], generic_dna).translate())


def hamming_distance(seq1, seq2):
    '''Hamming distance between two sequences of equal length'''
    return sum(x != y for x, y in zip(seq1, seq2))


with open('selection_sim_lineage_tree.p', 'rb') as fh:
    tree = pickle.load(fh)


colTree = CollapsedTree(tree=tree, name=tree.name, allow_repeats=True)
tree = colTree.tree


LBI = {l.strip().split()[0]: float(l.strip().split()[2]) for l in open('FitnessInference/20180303_17-04-51/sequence_ranking_terminals.txt') if l[0] != '#'}


# Print tree in ASCII:
#print(tree)
# Root name and sequence:
#tree.name
#tree.sequence

# Traverse through the nodes and print their affinity (Kd):
### N.B. Smaller numberical value of affinity means higher fitness
#for node in tree.traverse():
#    print(node.Kd)

LONR_dict = dict()
LONR_list = list()
delta_Kd_list = list()
for node in tree.iter_descendants():
    #print node.frequency
    aa = translate(node.sequence)
    aa_parent = translate(node.up.sequence)
    node.add_feature('NS_dist', hamming_distance(aa, aa_parent))
    if node.is_leaf():
        dist2tip = 0
    else:
        dist2tip = min([hamming_distance(node.sequence, l.sequence) for l in node.iter_descendants() if l.is_leaf()])
    node.add_feature('dist2tip', dist2tip)
    node.add_feature('delta_Kd', (node.up.Kd - node.Kd))

    if node.name in LBI:
        node.add_feature('LBI', LBI[node.name])

    parent = node.up
    N_children = len(parent.get_children())
#    node_N_leaves = sum([1 for l in node.traverse()])
#    parent_N_leaves = sum([1 for l in parent.iter_descendants()])
    node_N_leaves = sum([l.frequency for l in node.traverse()])
    parent_N_leaves = sum([l.frequency for l in parent.iter_descendants()])

#    if node.is_leaf() or N_children <= 1:
    if N_children <= 1:
        continue
    LONR = np.log(float(node_N_leaves) / (float(parent_N_leaves - node_N_leaves) / float(N_children - 1)))
    node.add_feature('LONR', LONR)

    if node.NS_dist not in LONR_dict:
        LONR_dict[node.NS_dist] = [node.LONR]
    else:
        LONR_dict[node.NS_dist].append(node.LONR)

    if node.frequency > 0 and node.dist2tip < 3:
        LONR_list.append(round(node.LONR, 3))
        delta_Kd_list.append(node.delta_Kd)


LONR_syn = np.array([node.LONR for node in tree.iter_descendants() if hasattr(node, 'LONR') and node.NS_dist == 0])
LONR_syn_mean = np.mean(LONR_syn)
LONR_syn_std = np.std(LONR_syn)

for node in tree.iter_descendants():
    if hasattr(node, 'LONR'):
        node.add_feature('LONR_Zscore', (node.LONR - LONR_syn_mean) / LONR_syn_std)



print LONR_dict

print LONR_list
print delta_Kd_list

print [node.LONR_Zscore for node in tree.iter_descendants() if hasattr(node, 'LONR') and node.NS_dist == 0]
print [node.LONR_Zscore for node in tree.iter_descendants() if hasattr(node, 'LONR') and node.NS_dist > 0]
print [node.delta_Kd for node in tree.iter_descendants() if hasattr(node, 'LONR') and node.NS_dist > 0]

print '****************'
print [round(node.LONR_Zscore, 3) for node in tree.iter_descendants() if hasattr(node, 'LONR') and node.NS_dist > 0 and node.frequency > 0]
print [round(node.delta_Kd, 3) for node in tree.iter_descendants() if hasattr(node, 'LONR') and node.NS_dist > 0 and node.frequency > 0]
print '****************'
print [round(node.LBI, 3) for node in tree.iter_descendants() if hasattr(node, 'LBI') and node.NS_dist > 0 and node.frequency > 0]
print [round(node.delta_Kd, 3) for node in tree.iter_descendants() if hasattr(node, 'LBI') and node.NS_dist > 0 and node.frequency > 0]



print '****************'
print [round(node.LONR_Zscore, 3) for node in tree.iter_descendants() if hasattr(node, 'LONR') and hasattr(node, 'LBI') and node.NS_dist > 0 and node.frequency > 0]
print [round(node.LBI, 3) for node in tree.iter_descendants() if hasattr(node, 'LBI') and hasattr(node, 'LONR') and node.NS_dist > 0 and node.frequency > 0]


