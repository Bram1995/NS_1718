import cplex
import copy
import pandas as pd
import numpy as np
from dijkstrasalgoritm import dijkstra, graph_creator
from col_generation import col_generation
from datetime import datetime
import matplotlib.pyplot as plt
import xlsxwriter
startTime = datetime.now()

## Load data
xl = pd.ExcelFile("Input_AE4424_Ass1P1.xlsx")
dfs = {sheet: xl.parse(sheet) for sheet in xl.sheet_names}
new_to_column = dfs['Arcs'].From
new_from_column = dfs['Arcs'].To
cost_column = dfs['Arcs'].Cost
capacity_column = dfs['Arcs'].Capacity
arc_column = np.array(range(31, 61, 1))
dfs_new = pd.DataFrame({'Arc':arc_column,'To':new_to_column,'From':new_from_column,'Cost':cost_column,
                        'Capacity':capacity_column},index=None)
dfs['Arcs'] = pd.concat([dfs['Arcs'],dfs_new])
dfs['Arcs'] = dfs['Arcs'].reset_index(drop=True)


## Create sets
arcs = range(len(dfs['Arcs'].Arc))
origins = dfs['Arcs'].From
destinations = dfs['Arcs'].To
locations = pd.concat([dfs['Arcs'].From,dfs['Arcs'].To]).unique()
commodities = range(1, len(dfs['Commodities'].Commodity) + 1)
quantity = np.array(dfs['Commodities'].Quant)

## Create input matrices
graph = graph_creator(dfs)
A_ineq = np.zeros((len(arcs),len(commodities)))
C = []
for i in range(len(commodities)):
    path, C_i = dijkstra(graph, dfs['Commodities'].From[i], dfs['Commodities'].To[i])
    C.append(float(C_i*quantity[i]))
    for j in range(len(path)-1):
        index = dfs['Arcs'].index[(dfs['Arcs'].From == path[j]) & (dfs['Arcs'].To == path[j+1])]
        A_ineq[index,i] = 1*dfs['Commodities'].Quant[i]
A_eq = np.eye(len(commodities))
rhs_ineq = list(dfs['Arcs'].Capacity)
rhs_eq = [1]*len(commodities)

slack_names=[]
## Add slack variables where needed
for i in range(len(arcs)):
    if A_ineq.sum(1)[i] - rhs_ineq[i] > 0:
        s = np.zeros((len(arcs), 1))
        s[i] = -1
        A_ineq = np.hstack((A_ineq,s))
        A_eq = np.hstack((A_eq,np.zeros((len(commodities),1))))
        slack_names.append('s_' + dfs['Arcs'].From[i] + '_' + dfs['Arcs'].To[i])


## Run model with initial matrices
RMP = cplex.Cplex()
RMP.objective.set_sense(RMP.objective.sense.minimize)
RMP.variables.add(obj= C,
                  names= ['f_k' + str(c) + '_0' for c in commodities])
RMP.variables.add(obj= [1000]*len(slack_names),
                  names= slack_names)
constraints_ineq = list()
for i in range(len(arcs)):
    constraints_ineq.append([A_ineq[i,:].nonzero()[0].tolist(), A_ineq[i,A_ineq[i,:].nonzero()[0]].tolist()])
RMP.linear_constraints.add(
    lin_expr = constraints_ineq,
    senses = ['L']*len(arcs),
    rhs = rhs_ineq)
constraints_eq = list()
for i in range(len(commodities)):
    constraints_eq.append([A_eq[i,:].nonzero()[0].tolist(), A_eq[i,A_eq[i,:].nonzero()[0]].tolist()])
RMP.linear_constraints.add(
    lin_expr = constraints_eq,
    senses = ['E']*len(commodities),
    rhs = rhs_eq)
RMP.solve()
print("Solution status :", RMP.solution.get_status())
print("Cost            : {0:.5f}".format(RMP.solution.get_objective_value()))
print()
sol = RMP.solution.get_values()
obj = RMP.solution.get_objective_value()
pi = np.array(RMP.solution.get_dual_values()[:len(arcs)])
sig = np.array(RMP.solution.get_dual_values()[len(arcs):])

time0 = datetime.now() - startTime
## COLUMN GENERATION ------------------------------------------------------
original_graph = copy.deepcopy(graph)
commodity_order = list(range(1,len(commodities)+1))
sig_vect = np.copy(sig)
quantity_vect = np.copy(quantity)
sig_vect[0] = 10000  # to get the while loop started
delta = np.copy(A_ineq[:,0:len(commodities)])
delta[delta>0] = 1
c_ij = dfs['Arcs']['Cost'].values
new_cost = np.array([c_ij-pi])

solution_list = [obj]
column_list = [len(sol)]
time_list = [str(time0)]
k = 1
while (np.inner(new_cost,delta.transpose()) < sig_vect/quantity_vect).any():
    RMP, A_eq, A_ineq, com_added = col_generation(RMP, original_graph, pi, sig, k, A_ineq, A_eq,dfs)
    # solve model
    RMP.solve()
    print("Solution status :", RMP.solution.get_status())
    print("Cost            : {0:.5f}".format(RMP.solution.get_objective_value()))
    print(datetime.now() - startTime)
    sol = RMP.solution.get_values()
    obj = RMP.solution.get_objective_value()
    solution_list.append(obj)
    column_list.append(len(sol))
    time_list.append(str(datetime.now() - startTime))
    pi = np.array(RMP.solution.get_dual_values()[:len(arcs)])
    sig = np.array(RMP.solution.get_dual_values()[len(arcs):])

    delta = np.hstack((delta, A_ineq[:, -len(com_added):]))
    delta[delta > 0] = 1
    new_cost = np.array([c_ij - pi])
    commodity_order = commodity_order + list(com_added)
    sig_vect = sig[np.array(commodity_order) - 1]
    quantity_vect = np.append(quantity_vect, quantity[np.array(com_added) - 1])
    k += 1
RMP.write('rmp.lp')


##  Question 3: Which two routes capacity increased by one unit?
np.nonzero(pi)
shadow_price_arcs = pi[:int(0.5*len(pi))] + pi[-int(0.5*len(pi)):]
ind_two_arcs = shadow_price_arcs.argsort()[0:2]
print("The best two routes for capacity increase by 1 are between: ")
print(origins[ind_two_arcs[0]] + ' and ' + destinations[ind_two_arcs[0]])
print()
print("and between:")
print(origins[ind_two_arcs[1]] + ' and ' + destinations[ind_two_arcs[1]])



plt.plot(column_list,solution_list,'b.')

plt.title('Amount of columns VS OBJ value')
plt.suptitle('Problem 1.2')
plt.ylabel('Objective value')
plt.xlabel('Amount of columns (#DV)')


plt.show()

dual_variables = np.concatenate((pi, sig))

workbook = xlsxwriter.Workbook('solution.xlsx')
worksheet = workbook.add_worksheet()

for i,variable in enumerate(pi):
    worksheet.write(i,2,variable)
    worksheet.write(i, 0, dfs['Arcs']['From'][i])
    worksheet.write(i, 1, dfs['Arcs']['To'][i])

for i in range(len(sig)):
    worksheet.write(i,4,sig[i])
    worksheet.write(i,3,i+1)


workbook.close()

solutionbook = xlsxwriter.Workbook('solution2.xlsx')
worksheet2 = solutionbook.add_worksheet()

for i, variable in enumerate(sol):
    worksheet2.write(i, 1, variable)
    worksheet2.write(i, 0, RMP.variables.get_names()[i])

solutionbook.close()
## Dis did not work..
# for c in range(len(commodities)):
#     path_new, C_new = dijkstra(graph, dfs['Commodities'].From[c], dfs['Commodities'].To[c])
#     C_new = float(C_new)
#     if C_new < sig[c]/quantity[c]:
#         A_ineq_add = np.zeros(len(arcs))
#         for j in range(len(path_new)-1):
#             index = dfs['Arcs'].index[(dfs['Arcs'].From == path_new[j]) & (dfs['Arcs'].To == path_new[j+1])]
#             A_ineq_add[index] = 1*quantity[c]
#         row_index = list(A_ineq_add.nonzero()[0])
#         row_value = list(A_ineq_add[row_index])+[1]
#         row_index.append(len(arcs)+c)
#         RMP.variables.add(obj= [C_new],
#                           names=['f_k' + str(c) + '_' + str(k)],
#                           columns=[row_index,row_value])
