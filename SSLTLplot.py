from SSLTL import getstates
from graphviz import Digraph

# See
def makedotMDP(betas, Tsas, Ls, acceptingMECs, pi_sa, showlabels=True):
    S = getstates(Tsas)
    print('we are here')
    stateid = {s: 'sx{0}'.format(i) for (i, s) in enumerate(S)}

    def sname(s):
        if type(s) is tuple:
            return ','.join(map(str, s))
        else:
            return '{0}'.format(s)

    dot = Digraph()

    for s in S:
        saccepting = any((s in MEC) for MEC in acceptingMECs)
        if saccepting:
            nodecolor = 'red'
        else:
            nodecolor = 'black'

        if showlabels and s in Ls:
            if type(Ls[s]) is tuple:
                labels = ','.join(map(str, Ls[s]))
            else:
                labels = Ls[s]
            dot.node(stateid[s], '{0} {{{1}}}'.format(sname(s), labels, color=nodecolor))
        else:
            dot.node(stateid[s], sname(s), color=nodecolor)

    # for beta in
    for (s, val) in betas.items():
        if val > 0:
            betaid = 'betas{0}'.format(sname(s))
            dot.node(betaid, '', shape='none', height='.0', width='.0')
            dot.edge(betaid, stateid[s], '{0}'.format(val))

    for (T, val) in Tsas.items():
        if val > 0:
            s = T[0]
            a = T[1]
            if type(s) is tuple:
                s = s[0]

            if (s,a) in pi_sa and pi_sa[s,a] > 0:
                edgecolor = 'blue'
            else:
                edgecolor = 'black'

            dot.edge(stateid[T[0]], stateid[T[2]], '{0}:{1}'.format(T[1], val), color=edgecolor)

    return dot



def makedotMC(betas, Tss, Ls, acceptingMECs, showlabels=True):
    S = getstates(Tss)

    stateid = {s: 'sx{0}'.format(i) for (i, s) in enumerate(S)}

    def sname(s):
        if type(s) is tuple:
            return ','.join(map(str, s))
        else:
            return '{0}'.format(s)

    dot = Digraph()

    for s in S:
        saccepting = any((s in MEC) for MEC in acceptingMECs)
        if saccepting:
            nodecolor = 'red'
        else:
            nodecolor = 'black'

        if showlabels and s in Ls:
            if type(Ls[s]) is tuple:
                labels = ','.join(map(str, Ls[s]))
            else:
                labels = Ls[s]
            dot.node(stateid[s], '{0} {{{1}}}'.format(sname(s), labels, color=nodecolor))
        else:
            dot.node(stateid[s], sname(s), color=nodecolor)

    # for beta in
    for (s, val) in betas.items():
        if val > 0:
            betaid = 'betas{0}'.format(sname(s))
            dot.node(betaid, '', shape='none', height='.0', width='.0')
            dot.edge(betaid, stateid[s], '{0}'.format(val))

    for (T, val) in Tss.items():
        if val > 0:
            edgecolor = 'black'
            dot.edge(stateid[T[0]], stateid[T[1]], '{0}'.format(val), color=edgecolor)

    return dot

