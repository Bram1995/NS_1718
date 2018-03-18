import pandas as pd
import numpy as np
import copy

## Load data
xl = pd.ExcelFile("Input_AE4424_Ass1P2.xlsx")
dfs = {sheet: xl.parse(sheet) for sheet in xl.sheet_names}
dfs['Flight'] = dfs['Flight'].set_index('Flight Number')
flights = dfs["Flight"].index.values.tolist()

itin = dfs["Itinerary"]["Itin No."].tolist()
A_1 = np.zeros((len(flights), len(itin)))
p_index = [1]*len(A_1[0])
sig = [0]*len(dfs['Itinerary'].index.values)
DV_label_list = []
def col_generation(dfs, pi,sig, p_index, A_ineq, DV_label_list):
    ## Create sets
    flights = dfs["Flight"].index.values.tolist()
    recapture = dfs["Recapture Rate"].index.values
    plist = dfs['Recapture Rate']['From Itinerary'].tolist()
    rlist = dfs['Recapture Rate']['To Itinerary'].tolist()
    fareplist = dfs['Recapture Rate']["Fare 'From'"].tolist()
    farerlist = dfs['Recapture Rate']["Fare 'To' "].tolist()
    bprlist = dfs['Recapture Rate']["Recapture Rate"].tolist()

    for i in recapture:
        p = plist[i]
        r = rlist[i]
        fare_p = fareplist[i]
        fare_r = farerlist[i]
        bpr = bprlist[i]
        #find flights belonging to p
        p_flight_numbers = dfs['Itinerary'].loc[ p , 'Leg 1' : 'Leg 2'].tolist()
        p_flight_numbers = [x for x in p_flight_numbers if str(x) != 'nan']
        p_pi_list = []
        for j, flight_number in enumerate(p_flight_numbers):
            p_pi_list.append(pi[flights.index(flight_number)])
        p_total_pi = sum(p_pi_list)

        # find flights belonging to r
        r_flight_numbers = dfs['Itinerary'].loc[r, 'Leg 1': 'Leg 2'].tolist()
        r_flight_numbers = [x for x in r_flight_numbers if str(x) != 'nan']
        r_pi_list = []
        for j, flight_number in enumerate(r_flight_numbers):
            r_pi_list.append(pi[flights.index(flight_number)])
        r_total_pi = sum(r_pi_list)


        tpr = fare_p - bpr * fare_r - p_total_pi + bpr * r_total_pi - sig[p]
        if tpr < 0:
            column = np.zeros((len(flights), 1))
            for leg in p_flight_numbers:
                p_leg_ind = flights.index(leg)
                column[p_leg_ind] = 1
            for leg in r_flight_numbers:
                r_leg_ind = flights.index(leg)
                column[r_leg_ind] = -1*bpr
            A_ineq = np.c_[A_ineq,column]

            DV_label_list.append('t' + '_' + str(p) + '_' + str(r))
            p_index.append(p)

    return(p_index,DV_label_list,A_ineq)


# def col_generation(model, original_graph, pi, sig, k, A_ineq, A_eq, com_added=None):
#     # adjust graph with new costs
#     if com_added is None:
#         com_added = []
#     graph=copy.deepcopy(original_graph)
#     C_real=float(0)
#     for i in range(len(arcs)):
#         origin = dfs['Arcs'].From[i]
#         destination = dfs['Arcs'].To[i]
#         graph[origin][destination] = original_graph[origin][destination] - pi[i]
#     for c in range(len(commodities)):
#         path_new, C_new = dijkstra(graph, dfs['Commodities'].From[c], dfs['Commodities'].To[c])
#         if C_new < sig[c] / quantity[c]:
#             A_ineq_add = np.zeros([len(arcs),1])
#             A_eq_add = np.zeros([len(commodities),1])
#             A_eq_add[c] = 1
#             com_added.append(c+1)
#             for j in range(len(path_new) - 1):
#                 index = dfs['Arcs'].index[(dfs['Arcs'].From == path_new[j]) & (dfs['Arcs'].To == path_new[j + 1])]
#                 A_ineq_add[index] = 1 * quantity[c]
#                 C_real = float(C_real+dfs['Arcs'].Cost[index])
#             A_ineq = np.hstack((A_ineq, A_ineq_add))
#             A_eq = np.hstack((A_eq, A_eq_add))
#             model.variables.add(obj=[C_real*quantity[c]],
#                                 names=['f_k' + str(c+1) + '_' + str(k)])
#     model.linear_constraints.delete()
#     # Add ineq constraints
#     constraints_ineq = list()
#     for i in range(len(arcs)):
#         constraints_ineq.append([A_ineq[i, :].nonzero()[0].tolist(), A_ineq[i, A_ineq[i, :].nonzero()[0]].tolist()])
#     model.linear_constraints.add(
#         lin_expr=constraints_ineq,
#         senses=['L'] * len(arcs),
#         rhs=rhs_ineq)
#     # Add eq constraints
#     constraints_eq = list()
#     for i in range(len(commodities)):
#         constraints_eq.append([A_eq[i, :].nonzero()[0].tolist(), A_eq[i, A_eq[i, :].nonzero()[0]].tolist()])
#     model.linear_constraints.add(
#         lin_expr=constraints_eq,
#         senses=['E'] * len(commodities),
#         rhs=rhs_eq)
#     return model, A_eq, A_ineq, com_added
