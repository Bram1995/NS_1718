import cplex
import copy
import numpy as np
import pandas as pd
from col_generation import col_generation


## Load data
xl = pd.ExcelFile("Assignment 1B/Input_AE4424_Ass1P2.xlsx")
dfs = {sheet: xl.parse(sheet) for sheet in xl.sheet_names}
dfs['Flight'] = dfs['Flight'].set_index('Flight Number')

## Create sets
flights = dfs["Flight"].index.values.tolist()
itin = dfs["Itinerary"]["Itin No."].tolist()
Fare = dfs["Itinerary"]["Fare"].tolist()

## Make cplex  model
RMP = cplex.Cplex()
RMP.objective.set_sense(RMP.objective.sense.minimize)
RMP.variables.add(obj=Fare,
                  names=['t' + str(p) + '_x' for p in range(len(itin))])

## Add constraint set 1
rhs_1 = []
A_1 = np.zeros((len(flights), len(itin)))
for i in range(len(flights)):
    leg1_ind = dfs["Itinerary"].index[dfs["Itinerary"]["Leg 1"] == flights[i]].tolist()
    leg2_ind = dfs["Itinerary"].index[dfs["Itinerary"]["Leg 2"] == flights[i]].tolist()
    index_it = leg1_ind + leg2_ind
    A_1[i, index_it] = 1
    q = dfs['Flight'].loc[flights[i], 'Capacity']
    cap1 = int(sum(dfs['Itinerary'].loc[dfs['Itinerary']['Leg 1'] == flights[i], 'Demand'].tolist()))
    cap2 = int(sum(dfs['Itinerary'].loc[dfs['Itinerary']['Leg 2'] == flights[i], 'Demand'].tolist()))
    rhs_1.append(float(q - cap1 - cap2))
    RMP.linear_constraints.add(
        lin_expr=[[index_it, A_1[i, index_it].tolist()]],
        senses=['G'],
        rhs=[rhs_1[i]])

# ## Add constraint set 2
# A_2 = np.eye(len(itin))
# for i in range(len(itin)):
#     RMP.linear_constraints.add(
#         lin_expr=[[[i], [1]]],
#         senses=['L'],
#         rhs=[dfs["Itinerary"]["Demand"].tolist()[i]])
#


## Solve model
RMP.solve()
print("Solution status :", RMP.solution.get_status())
print("Cost            : {0:.5f}".format(RMP.solution.get_objective_value()))
print()
sol = RMP.solution.get_values()
obj = RMP.solution.get_objective_value()

pi = np.array(RMP.solution.get_dual_values()[:len(flights)])
sig = np.array(RMP.solution.get_dual_values()[len(flights):])
RMP.write('rmp.lp')



#
# ## COLUMN GENERATION ------------------------------------------------------
# original_graph = copy.deepcopy(graph)
# commodity_order = list(range(1, len(commodities) + 1))
# sig_vect = np.copy(sig)
# quantity_vect = np.copy(quantity)
# sig_vect[0] = 10000  # to get the while loop started
# delta = np.copy(A_ineq[:, 0:len(commodities)])
# delta[delta > 0] = 1
# c_ij = dfs['Arcs']['Cost'].values
# new_cost = np.array([c_ij - pi])
#
# k = 1
# while (np.inner(new_cost, delta.transpose()) < sig_vect / quantity_vect).any():
#     RMP, A_eq, A_ineq, com_added = col_generation(RMP, original_graph, pi, sig, k, A_ineq, A_eq)
#     # solve model
#     RMP.solve()
#     print("Solution status :", RMP.solution.get_status())
#     print("Cost            : {0:.5f}".format(RMP.solution.get_objective_value()))
#     print()
#     sol = RMP.solution.get_values()
#     obj = RMP.solution.get_objective_value()
#     pi = np.array(RMP.solution.get_dual_values()[:len(arcs)])
#     sig = np.array(RMP.solution.get_dual_values()[len(arcs):])
#
#     delta = np.hstack((delta, A_ineq[:, -len(com_added):]))
#     delta[delta > 0] = 1
#     new_cost = np.array([c_ij - pi])
#     commodity_order = commodity_order + list(com_added)
#     sig_vect = sig[np.array(commodity_order) - 1]
#     quantity_vect = np.append(quantity_vect, quantity[np.array(com_added) - 1])
#     k += 1
# RMP.write('rmp.lp')
