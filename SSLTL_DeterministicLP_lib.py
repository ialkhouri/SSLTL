#!/usr/bin/python
# ---------------------------------------------------------------------------
# File: qcpex1.py
# Version 12.8.0
# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2009, 2017. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""
Entering and optimizing a quadratically constrained problem.

To run from the command line, use

   python qcpex1.py

velasqueza@LambaStack:~$ cd /opt/ibm/ILOG/CPLEX_Studio128/cplex/python/2.7/x86-64_linux
"""

from __future__ import print_function

import cplex
import time
import os
import itertools
from datetime import datetime



# BEKIW IS THE LATEST

def qcpex1_new(S, numNodes, init_sx, T_sas, Sx, Tx_sas, R_sas, MEC, acceptingMEC, Bx, Cx, SSconstraints, JKlist,
               rho=None, solutionType=0, epsilon=1e-6):
    # Notes:
    # 1) for T_sas and Tx_sas, if the transition sas is not in the dictionary, the this means the transition probability is 0.
    # 2) We get state-action pairs from T_sas. We do not assume a fixed number of actions for each state.

    numStates = len(S)
    numNodes = numNodes

    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    # number of accepting MECs:
    number_of_acceptingMECS = len(acceptingMEC)

    # Verify Tx_sas only contains unique values
    assert len(set(Tx_sas)) == len(Tx_sas)

    # Build sets and dictionaries we will need
    SUnknown = list(set(Sx).difference(set(Bx).union(set(Cx))))
    SA = {(s, a) for (s, a, s2) in T_sas}  # S X A
    As = {s: {a for (s2, a) in SA if s2 == s} for s in S}
    Qs = {s: {q for (s2, q) in Sx if s2 == s} for s in S}
    SxA = {(s, a) for (s, a, sp) in Tx_sas}
    # Asx = {s: {a for (s, a) in SxA} for s in Sx}
    SASx = list(itertools.product(SA, Sx))
    TG = {(s, s2) for ((s, a, s2), Tval) in Tx_sas.items() if s != s2 and Tval > 0.0}
    SK = {(s, k) for s in S for k in range(number_of_acceptingMECS)}

    Kunion = set()
    for (J, K) in JKlist:
        Kunion = Kunion.union(K)

    # Build dictionary of id's for each variable - we use a descriptive string for the id so that we can
    # export and read the LP that is built.
    z_id = {(s, q): 'z(s{0},q{1})'.format(s, q) for (s, q) in Sx}
    pi_id = {(s, a): 'pi(s{0},a{1})'.format(s, a) for (s, a) in SA}
    w_id = {(s, a, (sp, q)): 'w(s{0},a{1},(s{2},q{3}))'.format(s, a, sp, q) for ((s, a), (sp, q)) in SASx}
    x_id = {(s, a): 'x(s{0},a{1})'.format(s, a) for (s, a) in SxA}
    f_id = {((s, q), (sp, qp)): 'f((s{0},q{1}),(sp{2},qp{3}))'.format(s, q, sp, qp) for ((s, q), (sp, qp)) in TG}
    indicator_id = {(s, q): 'I(s{0},q{1})'.format(s, q) for (s, q) in Sx}
    indicatorS_id = {(s): 'Is(s{0})'.format(s) for (s) in S}
    indicatork_id = {(k): 'Ik(k{0})'.format(k) for (k) in range(number_of_acceptingMECS)}
    indicatorsk_id = {(s, k): 'Isk(s{0},k{1})'.format(s, k) for (s, k) in SK}

    all_var_ids = list(z_id.values()) + \
                  list(pi_id.values()) + \
                  list(w_id.values()) + \
                  list(x_id.values()) + \
                  list(f_id.values()) + \
                  list(indicator_id.values()) + \
                  list(indicatorS_id.values()) + \
                  list(indicatork_id.values()) + \
                  list(indicatorsk_id.values())

    def setproblemdata(p):
        # p.objective.set_sense(p.objective.sense.minimize)
        p.objective.set_sense(p.objective.sense.maximize)

        p.variables.add(names=list(z_id.values()), lb=[0.0] * len(z_id), ub=[1.0] * len(z_id))
        p.variables.add(names=list(w_id.values()), lb=[0.0] * len(w_id), ub=[1.0] * len(w_id))
        p.variables.add(names=list(x_id.values()), lb=[0.0] * len(x_id), ub=[1.0] * len(x_id))
        p.variables.add(names=list(f_id.values()), lb=[0.0] * len(f_id), ub=[1.0] * len(f_id))

        p.variables.add(names=list(pi_id.values()), lb=[0.0] * len(pi_id), ub=[1.0] * len(pi_id))
        p.variables.set_types([(i, p.variables.type.binary) for i in pi_id.values()])

        p.variables.add(names=list(indicator_id.values()), lb=[0.0] * len(indicator_id), ub=[1.0] * len(indicator_id))
        p.variables.set_types([(i, p.variables.type.binary) for i in indicator_id.values()])

        p.variables.add(names=list(indicatorS_id.values()), lb=[0.0] * len(indicatorS_id),
                        ub=[1.0] * len(indicatorS_id))
        p.variables.set_types([(i, p.variables.type.binary) for i in indicatorS_id.values()])

        p.variables.add(names=list(indicatork_id.values()), lb=[0.0] * len(indicatork_id),
                        ub=[1.0] * len(indicatork_id))
        p.variables.set_types([(i, p.variables.type.binary) for i in indicatork_id.values()])

        p.variables.add(names=list(indicatorsk_id.values()), lb=[0.0] * len(indicatorsk_id),
                        ub=[1.0] * len(indicatorsk_id))
        p.variables.set_types([(i, p.variables.type.binary) for i in indicatorsk_id.values()])

        p.variables.advanced.protect([i for i in all_var_ids])
        p.variables.advanced.tighten_lower_bounds([(i, 0.0) for i in all_var_ids])
        p.variables.advanced.tighten_upper_bounds([(i, 1.0) for i in all_var_ids])

        # # Objective function - OBJECTIVE MOTHER FU**AAAAAAAA
        # p.objective.set_linear(list(zip(list(z_id.values()), [1.0] * len(z_id))))
        lin_expr_vars = []
        lin_expr_vals = []
        for (s, q), a in SxA:
            for sp in S:
                if (s, a, sp) in T_sas and (s, a, sp) in R_sas:
                    lin_expr_vars.append(x_id[(s, q), a])
                    lin_expr_vals.append(T_sas[s, a, sp] * R_sas[s, a, sp])
        p.objective.set_linear(list(zip([0], [0])))  # (list(zip(list(lin_expr_vars),lin_expr_vals)))

        # old Objective function - OBJECTIVE MOTHER FU**AAAAAAAA
        # p.objective.set_linear(list(zip(list(z_id.values()), [1.0] * len(z_id))))

        # """ Constraint (i) """
        # p.linear_constraints.add(lin_expr=[cplex.SparsePair([z_id[init_sx[0],init_sx[1]]], val=[1.0])],
        #                              rhs=[1.0], senses=["E"],
        #                              names=['(i)_'])

        """ Constraint (ii) """
        """for s, q in SUnknown:
            lin_expr_vars = [z_id[s, q]]
            lin_expr_vals = [1]
            # We can iterate through Tx_sas, since if there is no entry in Tx_sas, this
            # means the transition probability is 0, and the corresponding x does not appear in the summation.
            for ((s2, q2), a, (sp, qp)), Tval in Tx_sas.items():
                if (s2, q2) == (s, q):
                    lin_expr_vars.append(w_id[s, a, (sp, qp)])
                    lin_expr_vals.append(-Tval)

            p.linear_constraints.add(lin_expr=[cplex.SparsePair(lin_expr_vars, val=lin_expr_vals)],
                                     senses=['E'], rhs=[0.0],
                                     names=["(ii)_"])"""

        """ Constraint (iii) """
        """for (s, a), (sp, qp) in SASx:
            p.linear_constraints.add(lin_expr=[cplex.SparsePair([w_id[s, a, (sp, qp)], pi_id[s, a]], val=[1.0, -1.0])],
                                     rhs=[0.0], senses=["L"],
                                     names=['(iii)_'])"""

        """ Constraint (iv) """
        """for (s, a), (sp, qp) in SASx:
            p.linear_constraints.add(lin_expr=[cplex.SparsePair([w_id[s, a, (sp, qp)], z_id[sp, qp]], val=[1.0, -1.0])],
                                     rhs=[0.0], senses=["L"],
                                     names=['(iv)_'])"""

        """" Constraint (v) """
        """for (s, a), (sp, qp) in SASx:
            p.linear_constraints.add(lin_expr=[cplex.SparsePair([w_id[s, a, (sp, qp)], z_id[sp, qp]], val=[1.0, -1.0])],
                                     rhs=[-1.0], senses=["G"],
                                     names=['(v)_'])"""

        """ Constraint (vi) """
        """p.linear_constraints.add(lin_expr=[cplex.SparsePair([z_id[s, q] for (s, q) in Bx], val=[1.0] * len(Bx))],
                                 senses=['E'], rhs=[len(Bx)],
                                 names=["(vi)_"])"""

        """ Constraint (vii) """
        """p.linear_constraints.add(lin_expr=[cplex.SparsePair([z_id[s, q] for (s, q) in Cx], val=[1.0] * len(Cx))],
                                 senses=['E'], rhs=[0.0],
                                 names=["(vii)_"])"""

        """ Constraint (ix) """
        for sp, qp in Sx:
            # We re-arrange the constraint as follows (this separates self loops into their own summation):
            #   (\sum_{(s,q) \in Sx, (s,q) \ne (s',q')} \sum_{a \in A(s)} x_sqa Tx((s',q')|(s,q),a)
            #        + \sum_{a \in A(s')} x_s'q'a (Tx((s',q')|(s',q'),a) - 1) == 0

            lin_expr_vars = []
            lin_expr_vals = []
            # Build the first summation - we can iterate through Tx_sas, since if there is no entry in Tx_sas, this
            # means the transition probability is 0, and the corresponding x does not appear in the summation.
            for ((s, q), a, (sp2, qp2)), Tval in Tx_sas.items():
                if (sp2, qp2) == (sp, qp) and (s, q) != (sp, qp):
                    lin_expr_vars.append(x_id[(s, q), a])
                    lin_expr_vals.append(Tval)
            # Build the second summation. Be careful: we need to include all possible actions, even if the transition probability is 0.
            for a in As[sp]:
                # Get transition probability - if not in dictionary, return 0
                Tval = Tx_sas.get(((sp, qp), a, (sp, qp)), 0.0)
                lin_expr_vars.append(x_id[(sp, qp), a])
                lin_expr_vals.append(Tval - 1.0)

            p.linear_constraints.add(lin_expr=[cplex.SparsePair(lin_expr_vars, val=lin_expr_vals)],
                                     senses=['E'], rhs=[0.0],
                                     names=["(ix)_"])

        """ Constraint (x) """
        lin_expr_vars = []
        for (s, q), a in SxA:
            lin_expr_vars.append(x_id[(s, q), a])

        p.linear_constraints.add(lin_expr=[cplex.SparsePair(lin_expr_vars, val=[1.0] * len(lin_expr_vars))],
                                 rhs=[1.0], senses=["E"],
                                 names=['(x)_'])

        """ Constraint (xi) """
        for (s, q), a in SxA:
            p.linear_constraints.add(lin_expr=[cplex.SparsePair([x_id[(s, q), a], pi_id[s, a]], val=[1.0, -1.0])],
                                     rhs=[0.0], senses=["L"],
                                     names=['(xi)_'])

        """ Constraint (viii) """
        for s in S:
            p.linear_constraints.add(lin_expr=[cplex.SparsePair([pi_id[s, a] for a in As[s]], val=[1.0] * len(As[s]))],
                                     rhs=[1.0], senses=["E"],
                                     names=['(viii)_'])

        """ Constraint (xii) """
        for (s, q), (sp, qp) in TG:
            lin_expr_vars = [f_id[(s, q), (sp, qp)]]
            lin_expr_vals = [1.0]
            for a in As[s]:
                if ((s, q), a, (sp, qp)) in Tx_sas:
                    lin_expr_vars.append(pi_id[s, a])
                    # We assume Tx>0 for any transition in TG
                    lin_expr_vals.append(-Tx_sas[(s, q), a, (sp, qp)])
            p.linear_constraints.add(lin_expr=[cplex.SparsePair(lin_expr_vars, val=lin_expr_vals)],
                                     rhs=[0.0], senses=["L"],
                                     names=['(xii)_'])

        """ Constraint (xiii) """
        for s, q in Sx:
            if (s, q) != init_sx:
                lin_expr_vars_lhs = [f_id[(sp, qp), (s2, q2)] for (sp, qp), (s2, q2) in TG if (s2, q2) == (s, q)]
                lin_expr_vals_lhs = [1.0] * len(lin_expr_vars_lhs)

                lin_expr_vars_rhs = [f_id[(s2, q2), (sp, qp)] for (s2, q2), (sp, qp) in TG if (s2, q2) == (s, q)]
                lin_expr_vals_rhs = [-1.0] * len(lin_expr_vars_rhs)

                lin_expr_vars_rhs += [indicator_id[s, q]]  # this is not a summation, its concatination
                lin_expr_vals_rhs += [-epsilon]

                p.linear_constraints.add(lin_expr=[
                    cplex.SparsePair(lin_expr_vars_lhs + lin_expr_vars_rhs, val=lin_expr_vals_lhs + lin_expr_vals_rhs)],
                                         rhs=[0.0], senses=["G"],
                                         names=['(xiii)_'])

        """ Constraint (xiv) """
        for s, q in Sx:
            lin_expr_vars = [f_id[(sp, qp), (s2, q2)] for (sp, qp), (s2, q2) in TG if (s2, q2) == (s, q)]
            lin_expr_vals = [1.0] * len(lin_expr_vars)
            lin_expr_vars.append(indicator_id[s, q])
            lin_expr_vals.append(-1)
            p.linear_constraints.add(lin_expr=[cplex.SparsePair(lin_expr_vars, val=lin_expr_vals)],
                                     rhs=[0.0], senses=["L"],
                                     names=['(xiv)_'])

        """ Constraint (xv) """
        for s, q in Sx:
            lin_expr_vars_lhs = [f_id[(s2, q2), (sp, qp)] for (s2, q2), (sp, qp) in TG if (s2, q2) == (s, q)]
            lin_expr_vals_lhs = [1.0] * len(lin_expr_vars_lhs)

            lin_expr_vars_rhs = [f_id[(sp, qp), (s2, q2)] for (sp, qp), (s2, q2) in TG if (s2, q2) == (s, q)]
            lin_expr_vals_rhs = [-0.5] * len(lin_expr_vars_rhs)

            p.linear_constraints.add(
                lin_expr=[
                    cplex.SparsePair(lin_expr_vars_lhs + lin_expr_vars_rhs, val=lin_expr_vals_lhs + lin_expr_vals_rhs)],
                rhs=[0.0], senses=["G"],
                names=['(xv)_'])

        """ Constraint (xvi) """
        for s, q in Sx:
            lin_expr_vars = [x_id[(s, q), a] for a in As[s]]
            lin_expr_vals = [1.0] * len(lin_expr_vars)
            lin_expr_vars.append(indicator_id[s, q])
            lin_expr_vals.append(-1)
            p.linear_constraints.add(lin_expr=[cplex.SparsePair(lin_expr_vars, val=lin_expr_vals)],
                                     rhs=[0.0], senses=["L"],
                                     names=['(xvi)_'])

        # """ Constraint (xvi) """
        # for s, q in Bx:
        #     lin_expr_vars_lhs = [x_id[(s, q), a] for a in As[s]]
        #     lin_expr_vals_lhs = [1.0] * len(lin_expr_vars_lhs)
        #
        #     lin_expr_vars_rhs = [f_id[(sp, qp), (s2, q2)] for (sp, qp), (s2, q2) in TG if (s2, q2) == (s, q)]
        #     lin_expr_vals_rhs = [-1.0] * len(lin_expr_vars_rhs)
        #
        #     p.linear_constraints.add(
        #         lin_expr=[cplex.SparsePair(lin_expr_vars_lhs + lin_expr_vars_rhs, val=lin_expr_vals_lhs + lin_expr_vals_rhs)],
        #         rhs=[0.0], senses=["L"],
        #         names=['(xvi)_'])

        """ Constraint (xvii) """
        for constr in SSconstraints:
            states = constr[0]
            sense = constr[1]
            bound = constr[2]

            lin_expr_vars = []
            for (s, q), a in SxA:
                if s in states:
                    lin_expr_vars.append(x_id[(s, q), a])
            lin_expr_vals = [1.0] * len(lin_expr_vars)

            p.linear_constraints.add(lin_expr=[cplex.SparsePair(lin_expr_vars, val=lin_expr_vals)],
                                     rhs=[bound], senses=[sense],
                                     names=['(xvii)_'])

        """ Constraint (xviii) """
        lin_expr_vars = []
        for (s, q), a in SxA:
            if q in Kunion:
                lin_expr_vars.append(x_id[(s, q), a])
        lin_expr_vals = [1.0] * len(lin_expr_vars)
        p.linear_constraints.add(lin_expr=[cplex.SparsePair(lin_expr_vars, val=lin_expr_vals)],
                                 rhs=[epsilon], senses=["G"],
                                 names=['(xviii)_'])

        """ Constraint (xix) """
        for k in range(number_of_acceptingMECS):
            lin_expr_vars = [indicatork_id[(k)]]
            lin_expr_vals = [-1.0]
            for (s, q) in set(acceptingMEC[k]):
                for a in As[s]:
                    lin_expr_vars.append(x_id[(s, q), a])
                    lin_expr_vals.append(1)
            p.linear_constraints.add(lin_expr=[cplex.SparsePair(lin_expr_vars, val=lin_expr_vals)],
                                     rhs=[0.0], senses=["L"],
                                     names=['(xix)_'])

        """ Constraint (xx) """
        for s in S:
            for k in range(number_of_acceptingMECS):
                lin_expr_vars = [indicatorsk_id[(s, k)]]
                lin_expr_vals = [-1.0]
                # print("HERE!!!")
                # print(number_of_acceptingMECS)

                for (s, q) in set(acceptingMEC[k]):
                    lin_expr_vars.append(indicator_id[(s, q)])
                    lin_expr_vals.append(1 / numNodes)

                p.linear_constraints.add(lin_expr=[cplex.SparsePair(lin_expr_vars, val=lin_expr_vals)],
                                         rhs=[0.0], senses=["L"],
                                         names=['(xx)_'])

        """ Constraint (xxi) """
        for s in S:
            lin_expr_vars = [indicatorS_id[(s)]]
            lin_expr_vals = [1.0]
            for k in range(number_of_acceptingMECS):
                lin_expr_vars.append(indicatorsk_id[(s, k)])
                lin_expr_vals.append(-1 / number_of_acceptingMECS)
                lin_expr_vars.append(indicatork_id[(k)])
                lin_expr_vals.append(1 / number_of_acceptingMECS)

            p.linear_constraints.add(lin_expr=[cplex.SparsePair(lin_expr_vars, val=lin_expr_vals)],
                                     rhs=[1.0], senses=["L"],
                                     names=['(xxi)_'])

        """ Constraint (xxii) """
        lin_expr_vars = []
        lin_expr_vals = []
        for s in S:
            # lin_expr_vals = [1.0] * len(lin_expr_vars)
            lin_expr_vars.append(indicatorS_id[s])
            lin_expr_vals.append(1)
        p.linear_constraints.add(lin_expr=[cplex.SparsePair(lin_expr_vars, val=lin_expr_vals)],
                                 rhs=[1.0], senses=["G"],
                                 names=['(xxii)_'])

        """ Constraint (xxiii) """
        for s in S:
            for k in range(number_of_acceptingMECS):
                lin_expr_vars = [indicatorsk_id[(s, k)]]
                lin_expr_vals = [1.0]
                # print("HERE!!!")
                # print(number_of_acceptingMECS)

                for (s, q) in set(acceptingMEC[k]):
                    lin_expr_vars.append(indicator_id[(s, q)])
                    lin_expr_vals.append(-1.0)

                p.linear_constraints.add(lin_expr=[cplex.SparsePair(lin_expr_vars, val=lin_expr_vals)],
                                         rhs=[0.0], senses=["L"],
                                         names=['(xxiii)_'])

    p = cplex.Cplex()
    startTime = time.time()
    setproblemdata(p)
    endTime = time.time()
    #p.write("{0}.lp".format(lpfilename))
    buildtime = endTime - startTime
    print("Done building model!!! It took " + str(buildtime) + " seconds!")
    # p.parameters.simplex.tolerances.feasibility = 1e-9
    # 4, 5: No negatives, 1: Only 1 negative
    p.parameters.lpmethod.set(solutionType)
    #p.parameters.simplex.tolerances.optimality.set(optTolerance)
    #p.parameters.simplex.tolerances.feasibility.set(feasTolerance)
    #p.parameters.mip.tolerances.integrality.set(intTolerance)
    print('test test test test for the jupyter')
    p.solve()

    solvetime = time.time() - endTime
    print("p.solve() took " + str(solvetime) + " seconds!")

    # solution.get_status() returns an integer code
    print("Solution status = ", p.solution.get_status(), ":", end=' ')
    # the following line prints the corresponding string
    solutionstatus = p.solution.status[p.solution.get_status()]
    print(solutionstatus)
    print("Solution value  = ", p.solution.get_objective_value())

    print()

    indicator = None

    # Build a dictionaries of solution values
    pi = {k: p.solution.get_values(id) for (k, id) in pi_id.items()}
    z = {k: p.solution.get_values(id) for (k, id) in z_id.items()}
    x_sqa = {k: p.solution.get_values(id) for (k, id) in x_id.items()}
    w = {k: p.solution.get_values(id) for (k, id) in w_id.items()}
    f = {k: p.solution.get_values(id) for (k, id) in f_id.items()}
    indicator = {k: p.solution.get_values(id) for (k, id) in indicator_id.items()}

    x_s = {}
    for s in S:
        x_s[s] = sum([sum([x_sqa[(s, q), a] for a in As[s]]) for q in Qs[s]])

    print('the propoer ILP is being used')
    return pi, z, x_sqa, x_s, w, f, indicator, solutionstatus, buildtime, solvetime
    #return pi, z, x_sqa, x_s, w, f, indicator, solutionstatus, buildtime, solvetime
    #return x_sqa, x_s,f, indicator, solutionstatus, buildtime, solvetime



