import pandas as pd
import numpy as np
from Assignment1A_dijkstrasalgoritm import dijkstra, graph_creator
import copy

def col_generation(model, original_graph, pi, sig, k, A_ineq, A_eq, dfs, com_added=None,):
    arcs = range(len(dfs['Arcs'].Arc))
    # origins = dfs['Arcs'].From
    # destinations = dfs['Arcs'].To
    # locations = pd.concat([dfs['Arcs'].From, dfs['Arcs'].To]).unique()
    commodities = range(1, len(dfs['Commodities'].Commodity) + 1)
    quantity = np.array(dfs['Commodities'].Quant)

    # adjust graph with new costs
    if com_added is None:
        com_added = []
    graph=copy.deepcopy(original_graph)
    C_real=float(0)
    for i in range(len(arcs)):
        origin = dfs['Arcs'].From[i]
        destination = dfs['Arcs'].To[i]
        graph[origin][destination] = original_graph[origin][destination] - pi[i]
    for c in range(len(commodities)):
        path_new, C_new = dijkstra(graph, dfs['Commodities'].From[c], dfs['Commodities'].To[c])
        if C_new < sig[c] / quantity[c]:
            A_ineq_add = np.zeros([len(arcs),1])
            A_eq_add = np.zeros([len(commodities),1])
            A_eq_add[c] = 1
            com_added.append(c+1)
            for j in range(len(path_new) - 1):
                index = dfs['Arcs'].index[(dfs['Arcs'].From == path_new[j]) & (dfs['Arcs'].To == path_new[j + 1])]
                A_ineq_add[index] = 1 * quantity[c]
                C_real = float(C_real+dfs['Arcs'].Cost[index])
            A_ineq = np.hstack((A_ineq, A_ineq_add))
            A_eq = np.hstack((A_eq, A_eq_add))
            model.variables.add(obj=[C_real*quantity[c]],
                                names=['f_k' + str(c+1) + '_' + str(k)])
    model.linear_constraints.delete()
    # Add ineq constraints
    constraints_ineq = list()
    for i in range(len(arcs)):
        constraints_ineq.append([A_ineq[i, :].nonzero()[0].tolist(), A_ineq[i, A_ineq[i, :].nonzero()[0]].tolist()])
    model.linear_constraints.add(
        lin_expr=constraints_ineq,
        senses=['L'] * len(arcs),
        rhs=list(dfs['Arcs'].Capacity))
    # Add eq constraints
    constraints_eq = list()
    for i in range(len(commodities)):
        constraints_eq.append([A_eq[i, :].nonzero()[0].tolist(), A_eq[i, A_eq[i, :].nonzero()[0]].tolist()])
    model.linear_constraints.add(
        lin_expr=constraints_eq,
        senses=['E'] * len(commodities),
        rhs= [1]*len(commodities))
    return model, A_eq, A_ineq, com_added
