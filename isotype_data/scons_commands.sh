

# Run a quick sweep through all isotype clonal families to find those that are challenging:
nohup scons --exp_data_test --quick --dnapars --iqtree --iqtree_option_str="-m 000000" --iqtree_option_str="-m 010010" --iqtree_option_str="-m 123450" --outdir=iso_sweep --xvfb --jobs=30 --exp_data=isotype_validation.csv --naiveIDexp=naive &> iso_sweep.log &

# Run full dnapars the challenging CFs:
nohup scons --exp_data_test --dnapars --outdir=iso_sweep_dnapars --xvfb --jobs=200 --srun --exp_data=isotype_validation_sweep.csv --naiveIDexp=naive &> iso_sweep_dnapars.log &



