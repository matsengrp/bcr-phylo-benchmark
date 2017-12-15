
# sweep = 'sweep_res.txt'
sweep = 'finalset.txt'

CFs = 'isotype_validation.csv'
# CFs_sweep = 'isotype_validation_sweep.csv'
CFs_sweep = 'isotype_validation_finalset.csv'

sweep_set = set()
with open(sweep) as fh:
    for l in fh:
        sweep_set.add(l.strip())

print 'Number of CFs after sweep:', len(sweep_set)


fho = open(CFs_sweep, 'w')
with open(CFs) as fh:
    fho.write(fh.next())
    for l in fh:
        if l.split(',')[0] in sweep_set:
            fho.write(l)

fho.close()

