print("hello world")

import cplex
import numpy as np
import matplotlib as mpl
import pandas as pd

xl = pd.ExcelFile("Input_AE4424_Ass1P1.xlsx")
dfs = {sheet: xl.parse(sheet) for sheet in xl.sheet_names}
print(dfs['Arcs'].Cost) # access DataFrame by sheet name
len(dfs['Commodities'].From.unique())
len(dfs['Commodities'].To.unique())

arcs = range(len(dfs['Arcs'].Arc))
origins = dfs['Arcs'].From
destinations = dfs['Arcs'].To
locations = pd.concat([dfs['Arcs'].From,dfs['Arcs'].To]).unique()
commodities = range(1, len(dfs['Commodities'].Commodity) + 1)


pandas.read_excel()

model=cplex.Cplex()
model.objective.set_sense(model.objective.sense.maximize)
model.variables.add(obj = [3,-1],names=['x1','x2'], lb=[0,0], types=[model.variables.type.integer,model.variables.type.integer])
model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=['x1','x2'],val=[3,-1])],senses=['L'],rhs=[2])

model.solve()
solution= model.solution
solution_array=solution.get_values()
model.write("lpex1.lp")
objective_value=solution.get_objective_value()

print(objective_value)
print(solution_array)
#print(solution.get_reduced_costs())
print(type(solution_array))
np.matrix=[]
