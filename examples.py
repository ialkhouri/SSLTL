import os
import time
import numpy as npss
import random
import math
import cplex
from SSLTL import productMDP_fromLTL,build_induced_MDP,print_productMDP_vars,build_induced_MC,buildgrid,getstates
#from SSLTL_DeterministicLP_lib import qcpex1,qcpex1_new,qcpex1_new_alvaro
from SSLTL_DeterministicLP_lib import qcpex1,qcpex1_new, qcpex1_new_alvaro
import spot
from statistics import mean, stdev
import cplex

ltl2drafolder = '/home/user/rabinizer4/bin'
SPOTbinfolder = '/home/user/spot-2.9/bin'
HOAfile = '/tmp/test.hoa'

from SSLTLplot import makedotMDP,makedotMC
from IPython.display import display # not needed with recent Jupyter




# Setup spot for fancy plots (it appears this is only recommended if the display command is run)
spot.setup()

# print('Current PYTHONPATH: {0}'.format(os.environ['PYTHONPATH']))

# Pending - need to include rewards
#R_sas = {(0,0,0):1 , (0,1,1):1}
R_sas ={}


SSconstraints=[]

# # ## SIMPLE EXAMPLE 1
# LTLstr = 'G F c'
# gridtype = '2x2'
# labelstr = (
# "aa"
# "ac"
# )


# #SIMPLE EXAMPLE 2
# LTLstr = 'G (F (c & X c))'
# gridtype = '2x2'
# labelstr = (
# "aa"
# "ac"
# )

# # SIMPLE EXAMPLE 3A
# LTLstr = 'GF(b & Xc & XXd)' # clockwise
# gridtype = '2x2'
# labelstr = (
# "ab"
# "dc"
# )


# # SIMPLE EXAMPLE 3B
# LTLstr = 'GF(d & Xc & XXb)' # counter-clockwise
# gridtype = '2x2'
# labelstr = (
# "ab"
# "dc"
# )


##Full example 1A: without steady-state constraints
LTLstr = 'GF(c & Xc)'
gridtype = '8x8'
labelstr = (
"aaaaaaaa"
"abbccbba"
"abbccbba"
"acccccca"
"acccccca"
"abbccbba"
"abbccbba"
"aaaaaaaa"
)


# # Full example 1B: with steady-state constraints
# LTLstr = 'GF(c & Xc)'
# gridtype = '8x8'
# labelstr = (
# "aaaaaaaa"
# "abbccbba"
# "abbccbba"
# "acccccca"
# "acccccca"
# "abbccbba"
# "abbccbba"
# "aaaaaaaa"
# )
# SSconstraints = [([25], 'G', 0.1)]

# ##Full example 1C: complicated
# LTLstr = 'GF(a & Xb & XXc & XXXc)'#/home/user/usr/lib/python3.7/site-packages
# gridtype = '8x8'
# labelstr = (
# "aaaaaaaa"
# "abbccbba"
# "abbccbba"
# "acccccca"
# "acccccca"
# "abbccbba"
# "abbccbba"
# "aaaaaaaa"
# )
# SSconstraints = [([63], 'G', 0.01),
#                  ([48,49,56,57], 'G', 0.01),
#                  ([17], 'G', 0.01),
#                  ([23], 'G', 0.01)]




betas = {0: 1}

startt = time.time()
T,Tdir,stategrid = buildgrid(1, gridtype, 1, 0)
print('Grid build time: {0} s'.format(time.time() - startt))

S = getstates(T)

Ls = {k: v for (k,v) in enumerate(labelstr)}

print(T)

key = list(T.keys())

#R_sas = T


(Sx,betax_s,Tx_sas,Lx_s,JKlist, MECs,acceptingMECs, BX,CX, DRAinfo) = productMDP_fromLTL(ltl2drafolder,SPOTbinfolder, betas,T,Ls, LTLstr, HOAfile, False)
display(DRAinfo['SPOTaut'])

# Currently, the LP needs an initial state - derive this from beta
betax_s_positive = [s for s, prob in betax_s.items() if prob > 0]
assert len(betax_s_positive) == 1
init_sx = betax_s_positive[0]


numNodes = len(list(DRAinfo.values())[1])



#x_sqa, x_s,f, indicator, solutionstatus, buildtime, solvetime = \


# pi, z, x_sqa, x_s, w, f, indicator, solutionstatus, buildtime, solvetime = \
#     qcpex1_new_alvaro(S, numNodes, init_sx, T, Sx, Tx_sas, R_sas, MECs, acceptingMECs, BX, CX, SSconstraints, JKlist,
#                rho=None, solutionType=0, epsilon=1e-4, optTolerance=1e-4, feasTolerance=1e-4, intTolerance=1e-4)

pi, z, x_sqa, x_s, w, f, indicator, solutionstatus, buildtime, solvetime = \
    qcpex1_new(S, numNodes, init_sx, T, Sx, Tx_sas, R_sas, MECs, acceptingMECs, BX, CX, SSconstraints, JKlist,
               rho=None, solutionType=0, epsilon=1e-6)




print('Build time: {0}'.format(buildtime))
print('Solve time: {0}'.format(solvetime))

numStates = len(S)

print('acceptingMECs = ' , acceptingMECs)



# ############################################################################ IF YOU ARE NOT USING ILP_NEW_2 keep, BELOW
# # ##########################################################for the case of ILP_new_2, get pi from below
# ###########################################################################################################
# SA = {(s, a) for (s, a, s2) in T}  # S X A
# As = {s: {a for (s2, a) in SA if s2 == s} for s in S}
# Qs = {s: {q for (s2, q) in Sx if s2 == s} for s in S}
#
# # define initial values for pi_sa
#
# pi_sa = [[None for j in range(4)] for i in range(len(S))]
# x_sa_from_sum_q = {}
# for s in S:
#     for a in As[s]:
#         x_sa_from_sum_q[(s,a)] = sum([x_sqa[(s, q), a] for q in Qs[s]])
#
#
# x_s = {}
# for s in S:
#     x_s[s] = sum([sum([x_sqa[(s, q), a] for a in As[s]]) for q in Qs[s]])
#
#
# for s in S:
#     for a in range(4):
#         pi_sa[s][a] = x_sa_from_sum_q[(s,a)] / x_s[s]
#
#
# print("pi_sa = ")
# for s in S:
#     for a in As[s]:
#         #print()
#         print('state, action = ',s,',',a,' prob = ' ,pi_sa[s][a])
#
# print(pi_sa)
#
# print('break')
####################################################################################################




##########################################################################################
########################################################################################
###################### UNCOMMENT BELOW IF YOU ARE USING THE OLD ILP OR ILP_NEW or ILP_new_alvaro

SA = {(s, a) for (s, a, s2) in T}  # S X A
As = {s: {a for (s2, a) in SA if s2 == s} for s in S}
Qs = {s: {q for (s2, q) in Sx if s2 == s} for s in S}

# define initial values for pi_sa

pi_sa = [[None for j in range(4)] for i in range(len(S))]

for s in S:
    for a in range(4):
        pi_sa[s][a] = pi[s, a]
print(pi_sa)

pi_arrows = ['<', 'v', '>', '^']

gridsize = (len(stategrid), len(stategrid[0]))
pi_mat = [[' ' for j in range(gridsize[1])] for i in range(gridsize[0])]
for i in range(gridsize[0]):
    for j in range(gridsize[1]):
        s = stategrid[i][j]
        pi_pos = [i for i,val in enumerate(pi_sa[s]) if val > 0]
        assert len(pi_pos) <= 1
        if pi_pos:
            pi_mat[i][j] = pi_arrows[pi_pos[0]]

for i in range(gridsize[0]):
    for j in range(gridsize[1]):
        print(pi_mat[i][j], end='')
    print('')

for i in range(gridsize[0]):
    for j in range(gridsize[1]):
        print("{0:3d}".format(stategrid[i][j]), end='')
    print('')



#display( makedotMDP(betas,T,Ls, frozenset(), {}, True) )
#display( makedotMDP(betax_s,Tx_sas,Lx_s, acceptingMECs, pi, False) )