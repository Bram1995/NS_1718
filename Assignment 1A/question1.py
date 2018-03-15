import cplex
import pandas as pd
import numpy as np

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
origins =dfs['Arcs'].From
destinations = dfs['Arcs'].To
locations = pd.concat([dfs['Arcs'].From,dfs['Arcs'].To]).unique()
commodities = range(1, len(dfs['Commodities'].Commodity) + 1)

## Create A_eq, A_ineq
A = np.zeros((len(locations),len(arcs)))
j = 0
for i in locations:
    index_from = dfs['Arcs'].loc[dfs['Arcs'].From == i].Arc -1
    index_to = dfs['Arcs'].loc[dfs['Arcs'].To == i].Arc - 1
    A[j, index_from] = 1
    A[j, index_to] = -1
    j += 1

def diag_block_mat_slicing(L):
    shp = L[0].shape
    N = len(L)
    r = range(N)
    out = np.zeros((N,shp[0],N,shp[1]),dtype=int)
    out[r,:,r,:] = L
    return out.reshape(np.asarray(shp)*N)

A_eq = diag_block_mat_slicing(((A,)*len(commodities)))

A_ineq = np.eye(len(arcs))
A_ineq = np.tile(A_ineq,len(commodities))

## Create RHS_eq, RHS_ineq
rhs_eq = np.empty((len(commodities)*len(locations),0))
for n in range(0,len(commodities)):
        # find to and from of commodity
        origin = dfs['Commodities'].From[n]
        destination = dfs['Commodities'].To[n]
        quantity = dfs['Commodities'].Quant[n]
        rhs = np.zeros((len(locations), 1))
        for i in range(0, len(locations)):
            if locations[i] == origin:
                rhs[i] = quantity
                #print(rhs[i])
            if locations[i] == destination:
                rhs[i] = -quantity
        rhs_eq = np.append(rhs_eq, rhs)

rhs_eq = list(rhs_eq)
rhs_ineq = list(dfs['Arcs'].Capacity)

## Create objective coefficients
C = dfs['Arcs'].Cost.tolist()
C = C*len(commodities)


## CPLEX
# Initialize the model
model = cplex.Cplex()

# Minimize objective function
model.objective.set_sense(model.objective.sense.minimize)

# Add variables
model.variables.add(obj= C,
                    names=
                    ['x' + '_' + dfs['Arcs'].From[i] + '_' + dfs['Arcs'].To[i] + '_' + str(c)
                     for i in arcs
                     for c in commodities
                     ],
                    )

# Add eq constraints
constraints_eq = list()
for i in range(len(commodities)*len(locations)):
    constraints_eq.append([A_eq[i,:].nonzero()[0].tolist(), A_eq[i,A_eq[i,:].nonzero()[0]].tolist()])

model.linear_constraints.add(
    lin_expr = constraints_eq,
    senses = ['E']*len(commodities)*len(locations),
    rhs = rhs_eq)


# Add ineq constraints
constraints_ineq = list()
for i in range(len(arcs)):
    constraints_ineq.append([A_ineq[i,:].nonzero()[0].tolist(), A_ineq[i,A_ineq[i,:].nonzero()[0]].tolist()])

model.linear_constraints.add(
    lin_expr = constraints_ineq,
    senses = ['L']*len(arcs),
    rhs = rhs_ineq)


# Solve the problem
model.solve()

# write out the model in LP format for debugging
model.write("q1.lp")
sol = model.solution.get_values()
obj = model.solution.get_objective_value()
pi = model.solution.get_dual_values()






