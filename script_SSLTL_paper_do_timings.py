#%%

import os
import time
import numpy as np
import math
import cplex
from SSLTL import productMDP_fromLTL,build_induced_MDP,print_productMDP_vars,build_induced_MC,buildgrid,getstates
from SSLTL_DeterministicLP_lib import qcpex1
import spot
from statistics import mean, stdev
import sys

ltl2drafolder = '~/rabinizer4/bin'
SPOTbinfolder = '~/usr/bin'
HOAfile = '/tmp/test.hoa'

from SSLTLplot import makedotMDP,makedotMC
from IPython.display import display # not needed with recent Jupyter

# Setup spot for fancy plots (it appears this is only recommended if the display command is run)
spot.setup()

print('Current PYTHONPATH: {0}'.format(os.environ['PYTHONPATH']))

#%%

# Pending - need to include rewards
R_sas = {}

LTLstr = '!danger U tool'

ntrials = 100
gridtypes = ['4x4']

solvetimes = [[None for j in range(ntrials)] for i in range(len(gridtypes))]
for gridnum, gridtype in enumerate(gridtypes):
    solvetimes.append([])

    i = 0
    while i < ntrials:
        print('\n\nGrid {0}, Iteration {1}\n'.format(gridtype, i+1))

        startt = time.time()
        T,Tdir,stategrid = buildgrid(1, gridtype, 0.8, 0.1)
        print('Grid build time: {0} s'.format(time.time() - startt))

        S = getstates(T)

        labelstates = list(np.random.choice(np.arange(0, len(S)), 2, replace=False))
        Ls = {labelstates[0]: "danger",
              labelstates[1]: "tool"}
        print('Labels: {0}'.format(Ls))

        betas = {0: 1}

        constraintstates = list(np.random.choice(np.arange(0, len(S)), int(math.sqrt(len(S))), replace=False))
        # SSconstraints = [(constraintstates, 'G', 0.75)]
        SSconstraints = [(constraintstates, 'G', 0.75)]
        print('SS states: {0}'.format(constraintstates))

        (Sx,betax_s,Tx_sas,Lx_s,JKlist, MECs,acceptingMECs, BX,CX, DRAinfo) = productMDP_fromLTL(ltl2drafolder,SPOTbinfolder, betas,T,Ls, LTLstr, HOAfile, False)
        #display(DRAinfo['SPOTaut'])

        beatx_s_positive = [s for s, prob in betax_s.items() if prob > 0]
        assert len(beatx_s_positive) == 1
        init_sx = beatx_s_positive[0]

        # v, pi, xProd, pr, buildtime, solvetimes[i] = qcpex1(S, betas, T, Sx, betax_s, Tx_sas, MECs, acceptingMECs, BX, CX, SSfunc, initialState = 0)
        try:
            pi, z, x_sqa, x_s, w, f, indicator, solutionstatus, buildtime, solvetime = qcpex1(S, init_sx, T, Sx, Tx_sas, R_sas, MECs, acceptingMECs, BX, CX, SSconstraints, JKlist, epsilon = 1e-6)
        except:
            print('Exception {0} in LP - skipping'.format(sys.exc_info()[0]))
        else:
            solvetimes[gridnum][i] = solvetime
            i += 1

#%%

for gridnum, gridtype in enumerate(gridtypes):
    print('{0} Solve time: {1}({2})'.format(gridtype, mean(solvetimes[gridnum]), stdev(solvetimes[gridnum])))

