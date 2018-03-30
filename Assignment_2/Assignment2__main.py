import cplex
import numpy as np
import pandas as pd

def col_generation(RMP,dfs, pi,sig_vect, p_index_list,vars_added):
    ## Create sets
    flights = dfs["Flight"].index.values.tolist()
    recapture = dfs["Recapture Rate"].index.values
    plist = dfs['Recapture Rate']['From Itinerary'].tolist()
    rlist = dfs['Recapture Rate']['To Itinerary'].tolist()
    fareplist = dfs['Recapture Rate']["Fare 'From'"].tolist()
    farerlist = dfs['Recapture Rate']["Fare 'To' "].tolist()
    bprlist = dfs['Recapture Rate']["Recapture Rate"].tolist()
    col_added = False
    for i in set(recapture)-set(vars_added):
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


        tpr = fare_p - bpr * fare_r - p_total_pi + bpr * r_total_pi - sig_vect[p]
        if round(tpr,5) < 0:
            col_added = True
            vars_added.append(i)
            cost = fare_p - bpr * fare_r
            p_leg_ind_list = []
            for leg in p_flight_numbers:
                p_index = flights.index(leg)
                p_leg_ind_list.append(p_index)

            r_leg_ind_list = []
            for leg in r_flight_numbers:
                r_index = flights.index(leg)
                r_leg_ind_list.append(r_index)

            same_flight_index = None
            for i in range(len(p_leg_ind_list)):
                if p_leg_ind_list[i] in r_leg_ind_list:
                    same_flight_index = p_leg_ind_list[i]
                    p_leg_ind_list = list(set(p_leg_ind_list)-{p_leg_ind_list[i]})
                    r_leg_ind_list = list(set(r_leg_ind_list) - {r_leg_ind_list[i]})
                    break
            p_index_list.append(p)
            if same_flight_index is not None:
                RMP.variables.add(obj=[cost],lb=[0],names=['t_'+str(p)+'_'+str(r)],columns=[[p_leg_ind_list+r_leg_ind_list + [same_flight_index],
                                                                             [1]*len(p_leg_ind_list) + [-bpr]*len(r_leg_ind_list) + [1 - bpr]]])
            else:
                RMP.variables.add(obj=[cost], lb=[0],names=['t_'+str(p)+'_'+str(r)], columns=[[p_leg_ind_list + r_leg_ind_list,
                [1] * len(p_leg_ind_list) + [-bpr] * len(r_leg_ind_list)]])

    return(RMP,p_index_list, col_added,vars_added)

## Load data
xl = pd.ExcelFile("Assignment_2/Assignment2.xlsx")
dfs = {sheet: xl.parse(sheet) for sheet in xl.sheet_names}
dfs['Flight'] = dfs['Flight'].set_index('Flight Number')
dfs['Flight'] = dfs['Flight'].fillna(10000000)  #fill in NAN values with big-M value

## Create sets
flights = dfs["Flight"].index.values.tolist()
itin = dfs["Itinerary"]["Itin No."].tolist()
Fare = dfs["Itinerary"]["Fare"].tolist()
Demand = dfs["Itinerary"]["Demand"].tolist()
actypes = dfs["Aircraft"]["Type"].tolist()

## Make cplex  model and add variables
RMP = cplex.Cplex()
RMP.objective.set_sense(RMP.objective.sense.minimize)
RMP.variables.add(obj=Fare,
                  names=['t' + str(p) + '_x' for p in range(len(itin))])
for i in flights:
    for k in actypes:
            RMP.variables.add(obj=[dfs["Flight"].loc[i,k]], names=['f_' + str(i) + '_' + str(k)])
## hier was ik: toevoegen variables y
RMP.variables.add(names=['y_' + str(k) + '_' + str(i) + '_' + str() for k in actypes for i in ])


## Add constraint set 1
for i in flights:
    RMP.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=['f_' + str(i) + '_' + str(k) for k in actypes],
                                   val=[1]*len(actypes))],
        senses=['E'],
        rhs=[1])

## Add constraint set 2
for i in flights:
    RMP.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=['f_' + str(i) + '_' + str(k) for k in actypes],
                                   val=[1]*len(actypes))],
        senses=['E'],
        rhs=[1])


## add seperate constraints for busses -> get own capacity
## Add constraint set 4
rhs_1 = []
A_1 = np.zeros((len(flights), len(itin)))
for i in range(len(flights)):
    leg1_ind = dfs["Itinerary"].index[dfs["Itinerary"]["Leg 1"] == flights[i]].tolist()
    leg2_ind = dfs["Itinerary"].index[dfs["Itinerary"]["Leg 2"] == flights[i]].tolist()
    index_it = leg1_ind + leg2_ind
    A_1[i, index_it] = 1
    cap = dfs['Flight'].loc[flights[i], 'Capacity']
    dem1 = int(sum(dfs['Itinerary'].loc[dfs['Itinerary']['Leg 1'] == flights[i], 'Demand'].tolist()))
    dem2 = int(sum(dfs['Itinerary'].loc[dfs['Itinerary']['Leg 2'] == flights[i], 'Demand'].tolist()))
    rhs_1.append(float(dem1 + dem2 - cap))
    RMP.linear_constraints.add(
        lin_expr=[[index_it, A_1[i, index_it].tolist()]],
        senses=['G'],
        rhs=[rhs_1[i]])


## Solve model
RMP.solve()
print("Solution status :", RMP.solution.get_status())
print("Cost            : {0:.5f}".format(RMP.solution.get_objective_value()))
print()
sol = np.array(RMP.solution.get_values())
sol_names = np.array(RMP.variables.get_names())
obj = RMP.solution.get_objective_value()

pi = np.array(RMP.solution.get_dual_values())
sig_it_index_list = []  # list of indices of itineraries for which row constraint is added
vars_added = []  # list of indices of recapture variables already added to model by column generation
sig_vect = np.array([0]*len(dfs['Itinerary']))  # initially we do not have sigma -> set to 0 for pricing problem
RMP.write('rmp.lp')


##  COLUMN & ROW GENERATION
Opt_Row = False
Opt_Col = False
p_index_list = np.arange(len(itin)).tolist()  #i row of p-numbers for all variables added
while Opt_Row is False or Opt_Col is False:
    # Column Generation loop
    while Opt_Col is False:
        col_added = False
        RMP, p_index_list, col_added, vars_added = col_generation(RMP, dfs, pi, sig_vect, p_index_list, vars_added)
        if col_added is True:
            RMP.solve()
            print("Cost            : {0:.5f}".format(RMP.solution.get_objective_value()))
            sol = np.array(RMP.solution.get_values())
            sol_names = np.array(RMP.variables.get_names())
            obj = RMP.solution.get_objective_value()
            pi = np.array(RMP.solution.get_dual_values()[:len(flights)])
            if len(np.array(RMP.solution.get_dual_values()[len(flights):])) != 0:
                sig_vect[sig_it_index_list] = np.array(RMP.solution.get_dual_values()[len(flights):])
            Opt_Row = False
        else:  # No column added, so can move to row generation
            Opt_Col = True

    # Row Generation loop
    while Opt_Row is False:
        row_added = False
        for p in range(len(itin)):
            if sum(sol[np.array(p_index_list)==p]) > Demand[p]:
                RMP.linear_constraints.add(
                    lin_expr=[[list(sol_names[np.array(p_index_list)==p]), [1]*sum(np.array(p_index_list)==p)]],
                    senses=['L'],
                    rhs=[Demand[p]])
                sig_it_index_list.append(p)
                row_added = True
        if row_added is True:
            RMP.solve()
            print("Cost            : {0:.5f}".format(RMP.solution.get_objective_value()))
            print("Solution status :", RMP.solution.get_status())
            sol = np.array(RMP.solution.get_values())
            sol_names = np.array(RMP.variables.get_names())
            obj = RMP.solution.get_objective_value()
            pi = np.array(RMP.solution.get_dual_values()[:len(flights)])
            sig_vect[sig_it_index_list] = np.array(RMP.solution.get_dual_values()[len(flights):])
            Opt_Col = False
        else:
            Opt_Row = True

RMP.write('rmp.lp')


# ##  Get flow per flightleg between hubs and division of itnierary pax
# df_hubflights = dfs['Flight'].loc[dfs['Flight']["ORG"].isin(["AEP","EZE"]) & dfs['Flight']["DEST"].isin(["AEP","EZE"])]
# for i in iterrows(df_hubflights):
#     flow[i] = Q_i - sum_r t_ir + sum_r b_ri t_ri

