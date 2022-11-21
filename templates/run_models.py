from multiprocessing import Pool
import numpy as np
import os

jobs = np.arange(40)+1
nthreads = 3
nzeros = 4

# Run the models in parallel
def run_model(jobnum):
    print('Running job'+jobnum)
    os.system('csh job'+jobnum)

p = Pool(nthreads)
jobnum = [str(x).zfill(nzeros) for x in jobs]
p.map(run_model, jobnum)
