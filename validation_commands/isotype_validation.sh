
# Dry run:
# nohup stdbuf -oL scons --dry-run --implicit-deps-unchanged --exp_data_test --gctree --dnapars --dnaml --igphyml --samm_rank --iqtree --iqtree_option_str="-m 000000" --iqtree_option_str="-m 010010" --iqtree_option_str="-m 123450" --outdir=isotype_validation --xvfb --jobs=10000 --srun --exp_data=sequence_data/isotype_validation_finalset.csv --naiveIDexp=naive &> isotype_validation.log &

nohup stdbuf -oL scons --exp_data_test --gctree --dnapars --dnaml --igphyml --samm_rank --iqtree --iqtree_option_str="-m 000000" --iqtree_option_str="-m 010010" --iqtree_option_str="-m 123450" --outdir=isotype_validation --xvfb --jobs=10000 --srun --exp_data=sequence_data/isotype_validation_finalset.csv --naiveIDexp=naive &> isotype_validation.log &


