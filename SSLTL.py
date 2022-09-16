# Depends on the tarjan package:
#   pip install tarjan

import os
from tarjan import tarjan
import spot
import itertools
import numpy as np




# ==== Miscellaneous

# Following function is based on code from https://www.geeksforgeeks.org/transitive-closure-of-a-graph/
# Prints transitive closure of graph[][] using Floyd Warshall algorithm
def transitiveClosure(T):
    """
    :param T: Transition matrix. T[i][j] is the probability of transitioning from state i to state j.
    :return: Reachability matrix reach. reach[i][j]==1 if node j is reachable from node i.
    """
    n = len(T)
    # reach = [i[:] for i in graph]
    reach = T
    '''Add all vertices one by one to the set of intermediate 
    vertices. 
     ---> Before start of a iteration, we have reachability value 
     for all pairs of vertices such that the reachability values 
      consider only the vertices in set  
    {0, 1, 2, .. k-1} as intermediate vertices. 
      ----> After the end of an iteration, vertex no. k is 
     added to the set of intermediate vertices and the  
    set becomes {0, 1, 2, .. k}'''
    for k in range(0, n):

        # Pick all vertices as source one by one
        for i in range(0, n):

            # Pick all vertices as destination for the
            # above picked source
            for j in range(0, n):
                # If vertex k is on a path from i to j,
                # then make sure that the value of reach[i][j] is 1
                reach[i][j] = reach[i][j] or (reach[i][k] and reach[k][j])

    # self.printSolution(reach)
    return reach




# ==== MDP helpers

def cull_transitions(S,betas,Tsas):
    done = False
    while not done:
        anychange = False

        for s in S:
            anyincoming = False
            if s in betas and betas[s] > 0:
                anyincoming = True

            for Tval in [Tval for (sas, Tval) in Tsas.items() if sas[2] == s]:
                if Tval > 0:
                    anyincoming = True

            if not anyincoming:
                S.remove(s)
                for key in [sas for sas in Tsas.keys() if sas[0] == s]:
                    del Tsas[key]
                    anychange = True

        if not anychange:
            done = True


def print_sorted_transitions(T):
    for (num,key) in enumerate(sorted(T)):
        print('  {0}: {1}{2}'.format(key, T[key], ',' if num<len(T)-1 else ''))

def build_induced_MDP(Sinduced, Tsas):
    Tsas_induced = {sas: val for (sas, val) in Tsas.items() if sas[0] in Sinduced and sas[2] in Sinduced}

    return Tsas_induced

def getstates(Tsas):
    return list(frozenset([sas[0] for sas in Tsas.keys()]))

def build_edge_preserve(S, Tsas):
    nstates = len(S)

    # Build a matrix which has a 1 if a transition exists.
    # We can tell which state the row/column represents
    # by looking at same position in "states" list (we don't assume
    # the states start at zero or are sequentially numbered).
    TchainEP = np.zeros([nstates, nstates])
    for key, value in Tsas.items():
        if value > 0:
            TchainEP[S.index(key[0]), S.index(key[2])] = 1

    return TchainEP



# def policy_to_dict(policy, statechar=False, actionchar=False):
#     T_pi_sa = {}
#     for (sidx,pi_s) in enumerate(policy):
#         if statechar:
#             stateid = 's{0}'.format(sidx)
#         else:
#             stateid = sidx
#         for (aidx,pi_sa) in enumerate(pi_s):
#             if actionchar:
#                 actionid = 'a{0}'.format(aidx)
#             else:
#                 actionid = aidx
#             T_pi_sa[stateid,actionid] = pi_sa
#
#     return T_pi_sa


def build_induced_MC(Tsas, pi_sa):
    Tss = {}
    for (sas, Tval) in Tsas.items():
        sfrom = sas[0]
        a = sas[1]
        sto = sas[2]

        slookup = sfrom
        if type(slookup) is tuple:
            slookup = slookup[0]
        pi_val = [pi_val for (pi_sa, pi_val) in pi_sa.items() if pi_sa[0] == slookup and pi_sa[1] == a]
        Tssval = Tval * sum(pi_val)
        if Tssval > 0:
            if (sfrom, sto) in Tss:
                Tss[sfrom, sto] += Tssval
            else:
                Tss[sfrom, sto] = Tssval

    return Tss






# ==== Automata

# == Atomic Property helpers

# Get index of atomic proposition in Sigma list
def apidx(Sigma,ap):
    try:
        idx = Sigma.index(ap)
    except ValueError:
        idx = -1

    if idx == -1:
        raise Exception('Atomic proposition {0} not found in list'.format(ap))

    return idx

# Given a list of atomic propositions (or a single atomic proposition),
# generate a list with truth values for each proposition in Sigma
def get_apflags(Sigma,aplist):
    if type(aplist) is not list:
        aplist = [aplist]

    apflags = [False] * len(Sigma)
    for ap in aplist:
        apflags[apidx(Sigma, ap)] = True

    return apflags

# Parse a transition string (from SPOT) to a make a set of flags:
#     1 - proposition should be present
#     0 - proposition should not be present
#    -1 - don't care (proposition not listed in string)
# NOTE: This parser was built ad-hoc and may NOT correctly handle all possible strings.
def parse_ap_prop(Sigma, propstr):
    if propstr.strip() == '1':
        apconds = [[-1] * len(Sigma)]
    else:
        apconds = []
        for expr in [p.strip() for p in propstr.split('|')]:
            if expr[0] == '(':
                expr = expr[1::]
            if expr[-1] == ')':
                expr = expr[:-1:]
            expr.strip()

            apconds_and = [-1] * len(Sigma)
            for expr_and in [p.strip() for p in expr.split('&')]:
                if expr_and[0] == '!':
                    flagval = 0
                    ap = expr_and[1:len(expr_and)]
                else:
                    flagval = 1
                    ap = expr_and

                apconds_and[apidx(Sigma, ap)] = flagval

            apconds.append(apconds_and)

    return apconds


# Evaluate the flags to get a truth value
def eval_ap(apconds, apflags):
    truth = False
    for apconds_and in apconds:
        truth_and = True
        for (apcond, apflag) in zip(apconds_and, apflags):
            if apcond != -1:
                if (apcond == 1 and not apflag) or \
                   (apcond == 0 and apflag):
                    truth_and = False
                    break

        if truth_and:
            truth = True
            break

    return truth


# Evaluate DRA delta function to get destination q
def eval_delta_function(Tqq, qsrc, apflags):
    Tqq_flag = Tqq[0]
    candidates = [(k[1], v) for (k, v) in Tqq_flag.items() if k[0] == qsrc]
    qdst = [cand[0] for cand in candidates if eval_ap(cand[1], apflags)]

    if not qdst:
        return None
    elif len(qdst) > 1:
        raise Exception('Multiple transitions have true propositions')
    else:
        return qdst[0]



def loaddra(hoa_filename, usestringid=True, Sigma=[]):
    a = spot.automaton(hoa_filename)

    if not a.prop_state_acc().is_true():
        raise Exception('Automaton must use state-based acceptance')

    # How to get the number of acceptance sets?
    num_fin_inf_pairs = a.acc().is_rabin()
    if num_fin_inf_pairs == -1:
        raise Exception('Automaton must be of Rabin type')

    Sigma_ltl = []
    for ap in a.ap():
        assert ap.is_leaf()
        Sigma_ltl.append(ap.ap_name())

    if not Sigma:
        Sigma = Sigma_ltl

    init_node = a.get_init_state_number()
    assert not a.is_univ_dest(init_node)  # not sure how to handle non-universal states

    bdict = a.get_dict()

    # The nodes can be given labels in the HOA file - can we access these from Python?
    nodenums = list(range(0, a.num_states()))
    if usestringid:
        Q = ['q{0}'.format(node) for node in nodenums]
    else:
        Q = nodenums

    q0 = Q[init_node]

    node_accept = {}
    Tqq_str = {}
    Tqq_flag = {}
    # stat = a.states()
    for (i, q) in zip(nodenums, Q):
        node_accept[q] = list(a.state_acc_sets(i).sets())

        for t in a.out(i):
            assert not a.is_univ_dest(i)  # not sure what this does, but we odn't support it

            srcq = Q[nodenums.index(t.src)]
            dstq = Q[nodenums.index(t.dst)]

            propstr = spot.bdd_format_formula(bdict, t.cond)
            Tqq_str[srcq, dstq] = propstr

            # Parse the proposition. This is an ugly way to do things. The "correct" way is probably
            # to work directly with the BDD structure built by SPOT. But, this will take time to understand.
            # Additionally, it appears the mapping between BDD id's and atomic propositions is not available
            # via the Python interface, and would require either writing C++ code to access, or modifying the library.

            #print('propstr: {0}'.format(propstr))
            Tqq_flag[srcq, dstq] = parse_ap_prop(Sigma, propstr)


    # Break up the "or" conditions in the acceptance condition and go through each component
    JKlist = []
    accept_top_disjuncts = a.acc().get_acceptance().top_disjuncts()
    assert len(accept_top_disjuncts) == num_fin_inf_pairs
    for acceptance in accept_top_disjuncts:
        # We will assume the fin and inf come in the proper order (fin comes before inf in the acceptance criteria
        inf_fin_mark_t = acceptance.used_inf_fin_sets()
        inf_fin_sets = [list(x.sets()) for x in inf_fin_mark_t]

        # We should only have one acceptance set inside the fin and inf
        assert len(inf_fin_sets[0]) == 1 and len(inf_fin_sets[1]) == 1

        JKlist.append((frozenset([j for (j, k) in node_accept.items() if inf_fin_sets[1][0] in k]),
                   frozenset([j for (j, k) in node_accept.items() if inf_fin_sets[0][0] in k])))

    return (a, Q,q0,Sigma_ltl,(Tqq_flag,Tqq_str),JKlist)

def LTL_to_DRA(ltl2drafolder, SPOTbinfolder, LTLstr, HOAfile):
    # We don't use the --annotations options with ltl2dra since it seems these are lost by autfilt
    cmd = '{0}/ltl2dra -i "{1}" | {2}/autfilt -S -C > "{3}"'.format(ltl2drafolder, LTLstr, SPOTbinfolder, HOAfile)
    errcode = os.system(cmd)
    if errcode != 0:
        raise Exception('Error code {0} when running command: {1}'.format(errcode, cmd))




# ==== Product MDP code

def build_productMDP(betas,Tsas,Sigma,Ls, Q,q0,Tqq):
    # Build state list (assumes each state has an outgoing edge or self loop)
    S = getstates(Tsas)

    # If no beta's, use a uniform distribution.
    if betas is None:
        betas = {s:1/len(S) for s in S}

    # Build ap flags from label set.
    # Make sure to cover every state (if no label specified, set flags to false),
    # as this makes things easier later.
    Ls_flag = {s:get_apflags(Sigma,aplist) for (s,aplist) in Ls.items()}
    for s in S:
        if s not in Ls_flag:
             Ls_flag[s] = [False] * len(Sigma)


    # BUILD PRODUCT MDP

    # Build states
    Sx = list(itertools.product(S, Q))

    # Build betax (zero entries may be omitted)
    betax_s = {}
    for (s, q) in Sx:
        # for (s,q) in [('s0','q0')]:
        # print('State:{0} q:{1} eval:{2}'.format(s,q,eval_delta(Tqq, q,Ls_flag[s])))
        if q == eval_delta_function(Tqq, q0, Ls_flag[s]) and s in betas:
            betax_s[(s, q)] = betas[s]
        # else:
        #    betaxs[(s,q)] = 0

    # Build transition matrix
    Tx_sas = {}
    for (sp, qp) in Sx:
        for q in Q:
            # for (s,q) in [('s0','q0')]:
            # print('sp:{0} qp:{1} q:{2} eval:{3}'.format(sp,qp,q,eval_delta_function(Tqq, q,Ls_flag[sp])))
            if qp == eval_delta_function(Tqq, q, Ls_flag[sp]):
                slist = [s for (s, q2) in Sx if q == q2]
                # print([(sas[0],sas[1]) for sas in Tsas.keys() if sas[0] in slist and sas[2]==sp])
                sa_candidates = [(sas[0], sas[1]) for sas in Tsas.keys() if sas[0] in slist and sas[2] == sp]
                for (s, a) in sa_candidates:
                    Tx_sas[(s, q), a, (sp, qp)] = Tsas[s, a, sp]

    # Build labels
    Lx_s = {(s, q): q for (s, q) in Sx}

    return (Sx, betax_s, Tx_sas, Lx_s)



def print_productMDP_vars(LTLstr, Q, initialState, betas,Tsas,Ls, betax_s,Tx_sas, MECs, acceptingMECs, BX,CX):
    print('# LTLstr = {0}'.format(LTLstr))


    print('\nnumautnodes = {0}'.format(len(Q)))
    print('Q = {0}\n'.format(Q))


    print('S = {0}'.format(getstates(Tsas)))

    print('initialState = {0}'.format(initialState))
    print('betas = {0}'.format(betas))

    print('Ls = {0}'.format(Ls))

    actions = list(frozenset([a for (_,a,_) in Tsas.keys()]))
    numactions = len(actions)
    print('\nnumactions = {0}'.format(numactions))
    print('actions = {0}'.format(actions))

    print('\nT = {')
    print_sorted_transitions(Tsas)
    print('}\n')


    print('Sx = {0}'.format(getstates(Tx_sas)))

    print('betax_s = {0}'.format(betax_s))

    print('\nTx_sas = {')
    print_sorted_transitions(Tx_sas)
    print('}\n')


    print('\nMEC = {0}'.format(MECs))
    print('\nacceptingMEC = {0}'.format(acceptingMECs))


    print('\nBx = {0}'.format(BX))
    print('Cx = {0}'.format(CX))





# ==== MECs

def find_accepting_MECs(MECs, JKlist):
    acceptingMECs = []
    for MEC in MECs:
        q_MEC = frozenset([val[1] for val in MEC])
        # print(q_MEC)

        for (Ji, Ki) in JKlist:
            # print(Ji)
            # print(Ki)
            # print(q_MEC.intersection(Ji))
            # print(q_MEC.intersection(Ki))
            if not q_MEC.intersection(Ji) and q_MEC.intersection(Ki):
                acceptingMECs.append(MEC)

    return frozenset(acceptingMECs)


def findMECs(Tsas):
    """
    Uses Algorithm 47 of [Baier and Katoen, 2008]
    :param Tsas: Dictionary with transitions.
    :return: Set with MECs
    """

    # Build dictionaries we will need for later

    S = frozenset([sas[0] for sas in Tsas.keys()])

    Act = {}
    for s in S:
        Act[s] = frozenset([sas[1] for sas in Tsas.keys() if sas[0] == s])

    Tss = {}
    Pre = {}
    Post = {}
    for s in S:
        Tss[s] = frozenset([sas[2] for sas in Tsas.keys() if sas[0] == s])
        Pre[s] = frozenset([(sas[0], sas[1]) for sas in Tsas.keys() if sas[2] == s])
        for a in Act[s]:
            Post[s, a] = frozenset([sas[2] for sas in Tsas.keys() if sas[0] == s and sas[1] == a])

    # Run Algorithm 47 of Principles of Model Checking by Baier,Katoen

    A = {}
    for s in S:
        A[s] = Act[s].copy()

    MEC = frozenset()
    MECnew = frozenset([S])

    while True:
        MEC = MECnew
        MECnew = frozenset()

        for T in MEC:
            # Find transitions for T
            TssTemp = {}
            for s in T:
                TssTemp[s] = frozenset([sas[2] for sas in Tsas.keys() if sas[0] == s and sas[1] in A[s]])

            R = frozenset()
            SCCs = frozenset([frozenset(SCC) for SCC in tarjan(TssTemp)])
            for SCC in SCCs.copy():
                if len(SCC) == 1:
                    s = list(SCC)[0]
                    if s not in TssTemp[s]:
                        SCCs = SCCs.difference(SCC) # Remove trivial SCCs

            for Ti in SCCs:
                for s in Ti:
                    A[s] = frozenset([alpha for alpha in A[s] if Post[s, alpha].issubset(Ti)])
                    if not A[s]:
                        R = R.union([s])

            while R:
                s = list(R)[0]
                R = R.difference([s])
                T = T.difference([s])

                PreT = frozenset([(t, beta) for (t, beta) in Pre[s] if t in T])
                for (t, beta) in PreT:
                    A[t] = A[t].difference([beta])
                    if not A[t]:
                        R = R.union([t])

            for Ti in SCCs:
                if T.intersection(Ti):
                    MECnew = MECnew.union([T.intersection(Ti)])
        if MEC == MECnew:
            break

    return MEC


def get_BX_CX(Tsas, acceptingMECs):
    S = getstates(Tsas)

    TchainEP = build_edge_preserve(S, Tsas)
    reach = transitiveClosure(TchainEP).astype(np.int)

    # meaning of unreachable dictionary: values cannot reach key
    unreachable = {s: [] for s in S}
    for froms in S:
        for tos in S:
            if reach[S.index(froms), S.index(tos)] == 0:
                unreachable[tos].append(froms)

    BX = frozenset()
    for acceptingMEC in acceptingMECs:
        BX = BX.union(acceptingMEC)

    CX = frozenset()
    for (tos, froms) in unreachable.items():
        if tos in BX:
            CX = CX.union(froms)

    return (BX,CX)



# This is the big function which puts everything together
def productMDP_fromLTL(ltl2drafolder,SPOTbinfolder, betas,Tsas,Ls, LTLstr, HOAfile, usestringqid=False):
    Sigma_Ls = list(set(Ls.values()))

    # Process LTL to DRA
    LTL_to_DRA(ltl2drafolder, SPOTbinfolder, LTLstr, HOAfile)
    aut, Q, q0, Sigma_ltl, Tqq, JKlist = loaddra(HOAfile, usestringqid, Sigma=Sigma_Ls)

    # We have two Sigmas (lists of atomic propositions) - one from the DRA and one from Ls.
    # We will use Sigma_Ls, since it should contain Sigma_ltl
    assert set(Sigma_ltl).issubset(set(Sigma_Ls))

    # Get a product MDP, removing unreachable states
    (Sx, betax_s, Tx_sas, Lx_s) = build_productMDP(betas, Tsas, Sigma_Ls, Ls, Q, q0, Tqq)
    cull_transitions(Sx, betax_s, Tx_sas)

    # Build MDPs
    MECs = list(findMECs(Tx_sas))
    acceptingMECs = list(find_accepting_MECs(MECs, JKlist))

    BX,CX = get_BX_CX(Tx_sas, acceptingMECs)

    DRAinfo = {'SPOTaut': aut,
               'Q': Q,
               'Sigma': Sigma_Ls,
               'Tqq': Tqq}

    return (Sx, betax_s, Tx_sas, Lx_s, JKlist, MECs, acceptingMECs, BX,CX, DRAinfo)



# ==== Examples
def buildgrid(version, gridtype, d, s):
    version = 1

    # d = 0.8
    # s = 0.1

    actiontemplate = {}

    if version == 1:
        # Vertical   -  MDP action
        # Horizontal -  MC state transition (induced by MDP action)

        #                     left  down  right  up   no action (self loop)

        # Unrestricted - middle of grid
        actiontemplate[1] = [[d, s, 0, s, 0],  # Move left
                             [s, d, s, 0, 0],  # Move down
                             [0, s, d, s, 0],  # Move right
                             [s, 0, s, d, 0]]  # Move up

        # ****
        # Left edge
        actiontemplate[2] = [[0, s, 0, s, d],  # Move left
                             [0, d, s, 0, s],  # Move down
                             [0, s, d, s, 0],  # Move right
                             [0, 0, s, d, s]]  # Move up

        # Top edge
        actiontemplate[3] = [[d, s, 0, 0, s],  # Move left
                             [s, d, s, 0, 0],  # Move down
                             [0, s, d, 0, s],  # Move right
                             [s, 0, s, 0, d]]  # Move up

        # Right edge
        actiontemplate[4] = [[d, s, 0, s, 0],  # Move left
                             [s, d, 0, 0, s],  # Move down
                             [0, s, 0, s, d],  # Move right
                             [s, 0, 0, d, s]]  # Move up

        # Bottom edge
        actiontemplate[5] = [[d, 0, 0, s, s],  # Move left
                             [s, 0, s, 0, d],  # Move down
                             [0, 0, d, s, s],  # Move right
                             [s, 0, s, d, 0]]  # Move up

        # ****
        # Top left corner
        actiontemplate[6] = [[0, s, 0, 0, d + s],  # Move left
                             [0, d, s, 0, s],  # Move down
                             [0, s, d, 0, s],  # Move right
                             [0, 0, s, 0, d + s]]  # Move up

        # Top right corner
        actiontemplate[7] = [[d, s, 0, 0, s],  # Move left
                             [s, d, 0, 0, s],  # Move down
                             [0, s, 0, 0, d + s],  # Move right
                             [s, 0, 0, 0, d + s]]  # Move up

        # Bottom left corner
        actiontemplate[8] = [[0, 0, 0, s, d + s],  # Move left
                             [0, 0, s, 0, d + s],  # Move down
                             [0, 0, d, s, s],  # Move right
                             [0, 0, s, d, s]]  # Move up

        # Bottom right corner
        actiontemplate[9] = [[d, 0, 0, s, s],  # Move left
                             [s, 0, 0, 0, d + s],  # Move down
                             [0, 0, 0, s, d + s],  # Move right
                             [s, 0, 0, d, s]]  # Move up

    else:
        raise Exception('unsupported')

    # ****
    # Empty - we can fill in these actions manually for special cases
    actiontemplate[99] = [[0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]]

    # See https://stackoverflow.com/questions/14681609/create-a-2d-list-out-of-1d-list
    def buildstategrid(n1,n2):
        return [list(range(i,i+n2)) for i in range(0,n1*n2,n2)]

    if gridtype == '2x2':
        stateype = [[6, 7],
                    [8, 9]]
        stategrid = buildstategrid(2,2)
    elif gridtype == '3x3':
        stateype = [[6, 3, 7],
                    [2, 1, 4],
                    [8, 5, 9]]
        stategrid = buildstategrid(3,3)

    elif gridtype == '4x4':
        stateype = [[6, 3, 3, 7],
                    [2, 1, 1, 4],
                    [2, 1, 1, 4],
                    [8, 5, 5, 9]]
        stategrid = buildstategrid(4,4)

    elif gridtype == '6x6':
        stateype = [[6, 3, 3, 3, 3, 7],
                    [2, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 4],
                    [8, 5, 5, 5, 5, 9]]
        stategrid = buildstategrid(6,6)

    elif gridtype == '8x8':
        stateype = [[6, 3, 3, 3, 3, 3, 3, 7],
                    [2, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 4],
                    [8, 5, 5, 5, 5, 5, 5, 9]]
        stategrid = buildstategrid(8,8)

    elif gridtype == '9x9':
        stateype = [[6, 3, 3, 3, 3, 3, 3, 3, 7],
                    [2, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 4],
                    [8, 5, 5, 5, 5, 5, 5, 5, 9]]
        stategrid = buildstategrid(9,9)

    elif gridtype == '12x12':
        stateype = [[6, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 7],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 9]]
        stategrid = buildstategrid(12,12)

    elif gridtype == '16x16':
        stateype = [[6, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 7],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 9]]
        stategrid = buildstategrid(16,16)

    elif gridtype == '24x24':
        stateype = [[6, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 7],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 9]]
        stategrid = buildstategrid(24,24)

    elif gridtype == '32x32':
        stateype = [[6, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 7],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4],
                    [8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 9]]
        stategrid = buildstategrid(32,32)
    else:
        assert False, 'Invalid grid type'



    def build_grid_Tsas_Tsad(actiontemplate, stategrid, statetype):
        Nactions = 4
        Ndirs = 5

        nextstatefunc = \
            {0: lambda ii, jj: stategrid[ii][jj - 1],
             1: lambda ii, jj: stategrid[ii + 1][jj],
             2: lambda ii, jj: stategrid[ii][jj + 1],
             3: lambda ii, jj: stategrid[ii - 1][jj],
             4: lambda ii, jj: stategrid[ii][jj]}

        Tsas = {}
        Tsad = {}
        for ii in range(0, len(stategrid)):
            for jj in range(0, len(stategrid[0])):
                s = stategrid[ii][jj]
                for mdpaction in range(0, Nactions):
                    for mctransition in range(0, Ndirs):
                        prob = actiontemplate[statetype[ii][jj]][mdpaction][mctransition]

                        assert (s, mdpaction, mctransition) not in Tsad
                        Tsad[s, mdpaction, mctransition] = prob

                        if prob > 0:
                            nextstate = nextstatefunc[mctransition](ii, jj)

                            assert (s, mdpaction, nextstate) not in Tsas
                            Tsas[s, mdpaction, nextstate] = prob

        return Tsas, Tsad

    Tsas, Tsad = build_grid_Tsas_Tsad(actiontemplate, stategrid, stateype)

    return Tsas, Tsad, stategrid


