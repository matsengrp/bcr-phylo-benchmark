
grep -Pv '^[\w\-]+\s+\d+\s+0\s+0\.0$' iso_sweep/*/isotype_validation.tsv | grep -v 'method' | cut -d '/' -f 2 | sort -un > sweep_res.txt

