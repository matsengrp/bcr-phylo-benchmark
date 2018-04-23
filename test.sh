# This is a minimal test to verify that all tools are setup the right way:
scons --simulate --igphyml --gctree --samm_rank --dnapars --dnaml --iqtree --iqtree_option_str="-m 000000" --iqtree_option_str="-m 010010" --iqtree_option_str="-m 123450" --outdir=neutral_sim_validation --xvfb --lambda=1.5 --lambda0=0.1825 --N=40 --nsim=2 --jobs=1 --random_naive="sequence_data/AbPair_naive_seqs.fa"
