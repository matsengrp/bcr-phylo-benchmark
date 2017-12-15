
# Grep out the clonal families that have misplaced isotypes in either "quick dnapars", JC, HKY or GTR:
grep -Pv '^[\w\-]+\s+\d+\s+0\s+0\.0$' iso_sweep/*/isotype_validation.tsv | grep -v 'method' | cut -d '/' -f 2 | sort -un > sweep_res.txt

# Get the set of clonal families that will run in dnapars in a reasonable (24h or less) time:
lt iso_sweep_dnapars/*/heavy/dnapars/dnapars_inferred_tree.tree | cut -d '/' -f 2 > finalset.txt


