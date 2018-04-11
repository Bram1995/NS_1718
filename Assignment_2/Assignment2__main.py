import cplex
import numpy as np
import pandas as pd
import copy
from datetime import datetime,timedelta,date
from datetime import time as dt
import time as tm

def col_generation(RMP,dfs, pi,sig_vect, p_index_list,vars_added, flights):
    ## Create sets
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
            p_pi_list.append(pi[constr_pi_begin+flights.index(flight_number)])
        p_total_pi = sum(p_pi_list)

        # find flights belonging to r
        r_flight_numbers = dfs['Itinerary'].loc[r, 'Leg 1': 'Leg 2'].tolist()
        r_flight_numbers = [x for x in r_flight_numbers if str(x) != 'nan']
        r_pi_list = []
        for j, flight_number in enumerate(r_flight_numbers):
            r_pi_list.append(pi[constr_pi_begin+flights.index(flight_number)])
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
bigM = 10000000
xl = pd.ExcelFile("Assignment_2\Assignment2.xlsx")
dfs = {sheet: xl.parse(sheet) for sheet in xl.sheet_names}
flight_dfs = dfs['Flight'].set_index('Flight Number')
flight_dfs.at[:,'Bus'] = bigM
actype_dfs = dfs['Aircraft'].set_index(['Type'])
actype_dfs.at['Bus',['Units','Seats','TAT (min)']] = [1,216,0]
flight_dfs = flight_dfs.fillna(bigM)  #fill in NAN values with big-M value
indices = flight_dfs.loc[(flight_dfs['ORG'] == 'AEP') & (flight_dfs['DEST'] == 'EZE') |
                         (flight_dfs['ORG'] == 'EZE') & (flight_dfs['DEST'] == 'AEP')].index.tolist()
flight_dfs.at[indices, actype_dfs.index.values.tolist()] = bigM  # set bigM for not possible assignments
flight_dfs.at[indices, actype_dfs.index.values.tolist()[-1]] = 4500.0  # set bus cost
flight_dfs.at["AR1361","B737"] = bigM


## Create sets
flights = flight_dfs.index.values.tolist()
itin = dfs["Itinerary"]["Itin No."].tolist()
Fare = dfs["Itinerary"]["Fare"].tolist()
Demand = dfs["Itinerary"]["Demand"].tolist()
actypes = actype_dfs.index[:4]  # excl bus
transport = actype_dfs.index  # all modes of transport, incl bus
airport_list = sorted(list(set(flight_dfs['ORG'])))
seats = actype_dfs['Seats'].values.tolist()

start_time = tm.time()
nodes_dfs_dict = {}  # dictionary with for every ACtype all nodes
action_dict = {}
t_cut = dt(0,9,0)
time_check_dict = {}

for i in actype_dfs.index[:4]:
    tat = int(actype_dfs.loc[i,'TAT (min)'])
    flight_dataframe = copy.deepcopy(flight_dfs)
    flight_dataframe = flight_dataframe.loc[(flight_dataframe[i] != bigM), ['ORG', 'DEST', 'Departure', 'Arrival', i]]
    flight_dataframe = flight_dataframe.loc[(flight_dataframe[i] != 0.), ['ORG', 'DEST', 'Departure', 'Arrival', i]]
    flight_dataframe_sub_1 = flight_dataframe.loc[:, ['ORG', 'Departure']].rename(columns={'ORG': 'Airport', 'Departure': 'Time'})
    flight_dataframe_sub_1.at[:, 'Action'] = 'DEP'
    flight_dataframe_sub_2 = flight_dataframe.loc[:, ['DEST', 'Arrival']].rename(columns={'DEST': 'Airport', 'Arrival': 'Time'})
    flight_dataframe_sub_2.at[:, 'Action'] = 'ARR'


    for k in flight_dataframe_sub_2.index:
        time = datetime.combine(date(1,1,1),flight_dataframe_sub_2.loc[k,'Time'])
        flight_dataframe_sub_2.at[k,'Time'] = (time + timedelta(minutes= tat)).time()
    # selecting flights where dep time is larger than
    time_check_dataframe = copy.deepcopy(flight_dataframe_sub_1)
    time_check_dataframe['Time ARR']=flight_dataframe_sub_2['Time']
    time_check_dataframe = time_check_dataframe.drop(columns=['Action','Airport']).rename(columns= {'Time':'Time DEP'})
    time_check_dataframe = time_check_dataframe.loc[((time_check_dataframe['Time ARR'] < time_check_dataframe['Time DEP']) &
                                                     (time_check_dataframe['Time ARR'] > t_cut)) | ((time_check_dataframe['Time DEP'] < t_cut)
                                                                                                    & (time_check_dataframe['Time ARR'] > t_cut))]
    time_check_dict[i] = copy.deepcopy(time_check_dataframe)
    nodes_dataframe = copy.deepcopy(pd.concat([flight_dataframe_sub_1, flight_dataframe_sub_2])).sort_values(
        ['Airport', 'Time'], ascending=[True, True])
    nodes_dataframe_action = copy.deepcopy(nodes_dataframe)
    nodes_dataframe = nodes_dataframe.drop_duplicates(subset=['Airport', 'Time'])
    nodes_dataframe = nodes_dataframe.drop(columns='Action')
    nodes_dfs_dict[i] = copy.deepcopy(nodes_dataframe.reset_index(drop=True))
    action_dict[i] = copy.deepcopy(nodes_dataframe_action)

ga_dict = {}  # dictionary with for every ACtype all ground arcs per airport
time_check_ga_dict = {}
GA_labels = []
for k in actype_dfs.index[:4]:
    time_check_ga_dict[k] = pd.DataFrame(columns=['Airport','Time1','Time2'])
    for i in airport_list:
        ground_arc_dfs = pd.DataFrame(columns = ['Airport','Time1','Time2'])
        arc_selection = nodes_dfs_dict[k].loc[nodes_dfs_dict[k]['Airport']==i,:]
        for p,j in enumerate(arc_selection.index):
            time1 = arc_selection.iloc[p, 1]
            if p == len(arc_selection.index)-1:
                time2 =  arc_selection.iloc[0, 1]
                GA_label = k + '_' + i + '_' + str(time1) + '_' + str(time2)
                index = p+1
                ground_arc_dfs.at[index, ['Airport','Time1', 'Time2']] = [i,time1, time2]
            else:
                time2 = arc_selection.iloc[p+1, 1]
                GA_label = k + '_' + i + '_' + str(time1) + '_' + str(time2)
                index = p+1
                ground_arc_dfs.at[index, ['Airport', 'Time1', 'Time2']] = [i,time1, time2]
            GA_labels.append(GA_label)
        ga_time_dfs = copy.deepcopy(ground_arc_dfs)
        ga_time_dfs = ga_time_dfs.loc[((ga_time_dfs['Time2'] < ga_time_dfs['Time1']) &
                                       (ga_time_dfs['Time2'] > t_cut)) | ((ga_time_dfs['Time1'] < t_cut)
                                                                          & (ga_time_dfs['Time2'] > t_cut))]
        ga_dict[k + '_' + i] = copy.deepcopy(ground_arc_dfs)
        time_check_ga_dict[k] = time_check_ga_dict[k].append(copy.deepcopy(ga_time_dfs), ignore_index=True)

##  Find time for cutting timespace network
#t_cut = dt(0,9,0)
# for k in actypes:
#     print(nodes_dfs_dict[k]["Time"].isin([t_cut]).any())

## Make cplex  model and add variables
RMP = cplex.Cplex()
RMP.objective.set_sense(RMP.objective.sense.minimize)
RMP.variables.add(obj=Fare,
                  names=['t' + str(p) + '_x' for p in range(len(itin))])
for i in flights:
    for k in actype_dfs.index:
        RMP.variables.add(obj=[flight_dfs.loc[i,k]], names=['f_' + str(i) + '_' + str(k)], ub=[1])
RMP.variables.add(names=['y_' + z for z in GA_labels])

## Add constraint set 1
for i in flights:
    RMP.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=['f_' + str(i) + '_' + str(k) for k in actype_dfs.index],
                                   val=[1]*len(actype_dfs.index))],
        senses=['E'],
        rhs=[1])

## Add constraint set 2
k = actypes[0]
n = nodes_dfs_dict[k].iloc[0]
for k in actypes:
    for nix, n in nodes_dfs_dict[k].iterrows():
        airport = n["Airport"]
        time = n["Time"]
        OI_set = action_dict[k].loc[(action_dict[k]["Time"] == time) & (action_dict[k]["Airport"] == airport)]
        O_set = OI_set.loc[(OI_set["Action"] == "DEP")]
        I_set = OI_set.loc[(OI_set["Action"] == "ARR")]
        time1 = ga_dict[k + '_' + airport].loc[ga_dict[k + '_' + airport]["Time2"] == time, "Time1"].values[0]
        time2 = ga_dict[k + '_' + airport].loc[ga_dict[k + '_' + airport]["Time1"] == time, "Time2"].values[0]
        # #  n+
        # print('y_' + k + '_' + airport + '_' + str(time) + '_' + str(time2))
        # #  n-
        # print('y_' + k + '_' + airport + '_' + str(time1) + '_' + str(time))
        # # f O +
        # print('f_' + i + '_' + k for i in O_set.index)
        # # f I -
        # print('f_' + i + '_' + k for i in I_set.index)
        RMP.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=['y_' + k + '_' + airport + '_' + str(time) + '_' + str(time2)] + [
                'y_' + k + '_' + airport + '_' + str(time1) + '_' + str(time)] + ['f_' + i + '_' + k for i in O_set.index] + ['f_' + i + '_' + k for i in I_set.index],
                                       val=[-1, 1] + [-1] * len(O_set) + [1] * len(I_set))],
            senses=['E'],
            rhs=[0])

## add constraint set 3: total amount of AC constraint
for k in actypes:
    flight_set = time_check_dict[k]
    ga_set = time_check_ga_dict[k]
    rhs = int(actype_dfs.loc[k,'Units'])
    RMP.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=['f_' + i + '_' + k for i in flight_set.index] + [
            'y_' + k + '_' + ga_set.loc[j,'Airport'] + '_' + str(ga_set.loc[j,'Time1']) + '_' + str(ga_set.loc[j,'Time2'])
            for j in ga_set.index], val=[1] * len(flight_set) + [1] * len(ga_set))], senses=['L'], rhs=[rhs])

constr_pi_begin = RMP.linear_constraints.get_num()

## Add constraint set 4
rhs_1 = []
A_1 = np.zeros((len(flights), len(itin)))
for ix, i in enumerate(flights):
    f_list = ['f_' + i + '_' + k for k in transport]
    leg1_ind = dfs["Itinerary"].index[dfs["Itinerary"]["Leg 1"] == i].tolist()
    leg2_ind = dfs["Itinerary"].index[dfs["Itinerary"]["Leg 2"] == i].tolist()
    index_it = leg1_ind + leg2_ind
    A_1[ix, index_it] = 1
    #cap = dfs['Flight'].loc[flights[i], 'Capacity']
    dem1 = int(sum(dfs['Itinerary'].loc[dfs['Itinerary']['Leg 1'] == i, 'Demand'].tolist()))
    dem2 = int(sum(dfs['Itinerary'].loc[dfs['Itinerary']['Leg 2'] == i, 'Demand'].tolist()))
    rhs_1.append(float(dem1 + dem2)) #-cap
    RMP.linear_constraints.add(
        lin_expr=[[f_list + index_it, seats + A_1[ix, index_it].tolist()]],
        senses=['G'],
        rhs=[rhs_1[ix]])

initial_num_var = RMP.variables.get_num()
initial_num_constr = RMP.linear_constraints.get_num()

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



##  COLUMN & ROW GENERATION
Opt_Row = False
Opt_Col = False
itcount = 0  # counter for number of iterations of the outer while loop
itcount_col = 0  # counter for number of iterations of the CG while loop
itcount_row = 0  # counter for number of iterations of the row gen while loop

p_index_list = np.arange(len(itin)).tolist()  #i row of p-numbers for all variables added
while Opt_Row is False or Opt_Col is False:
    # Column Generation loop
    while Opt_Col is False:
        col_added = False
        RMP, p_index_list, col_added, vars_added = col_generation(RMP, dfs, pi, sig_vect, p_index_list, vars_added, flights)
        if col_added is True:
            itcount_col += 1
            print("Added columns")
            RMP.solve()
            print("Cost            : {0:.5f}".format(RMP.solution.get_objective_value()))
            sol_t = np.array(RMP.solution.get_values(np.arange(len(itin)).tolist()) + RMP.solution.get_values()[len(itin)+len(GA_labels)+len(flights)*len(actype_dfs.index):])
            sol_names_t = np.array(RMP.variables.get_names(np.arange(len(itin)).tolist()) + RMP.variables.get_names()[len(itin)+len(GA_labels)+len(flights)*len(actype_dfs.index):])
            obj = RMP.solution.get_objective_value()
            pi = np.array(RMP.solution.get_dual_values())
            if len(np.array(RMP.solution.get_dual_values()[constr_pi_begin+len(flights):])) != 0:
                sig_vect[sig_it_index_list] = np.array(RMP.solution.get_dual_values()[constr_pi_begin+len(flights):])
            Opt_Row = False
        else:  # No column added, so can move to row generation
            Opt_Col = True

    # Row Generation loop
    while Opt_Row is False:
        row_added = False
        for p in range(len(itin)):
            if sum(sol_t[np.array(p_index_list)==p]) > Demand[p]:
                RMP.linear_constraints.add(
                    lin_expr=[[list(sol_names_t[np.array(p_index_list)==p]), [1]*sum(np.array(p_index_list)==p)]],
                    senses=['L'],
                    rhs=[Demand[p]])
                sig_it_index_list.append(p)
                row_added = True
        if row_added is True:
            itcount_row += 1
            print("Added rows")
            RMP.solve()
            print("Cost            : {0:.5f}".format(RMP.solution.get_objective_value()))
            print("Solution status :", RMP.solution.get_status())
            sol_t = np.array(RMP.solution.get_values(np.arange(len(itin)).tolist()) + RMP.solution.get_values()[len(itin) + len(GA_labels) + len(flights) * len(actype_dfs.index):])
            sol_names_t = np.array(RMP.variables.get_names(np.arange(len(itin)).tolist()) + RMP.variables.get_names()[len(itin) + len(GA_labels) + len(flights) * len(actype_dfs.index):])
            obj = RMP.solution.get_objective_value()
            pi = np.array(RMP.solution.get_dual_values())
            sig_vect[sig_it_index_list] = np.array(RMP.solution.get_dual_values()[constr_pi_begin+len(flights):])
            Opt_Col = False
        else:
            Opt_Row = True
    itcount += 1

## Make MILP and solve
for i in range(len(itin),len(itin) + len(flights) * len(actype_dfs.index)):
    RMP.variables.set_types(i,'B')
print("Transformed LP to MILP")
RMP.solve()
print("Cost            : {0:.5f}".format(RMP.solution.get_objective_value()))
print("Solution status :", RMP.solution.get_status())
Opt_Row = False

## Row Generation loop
while Opt_Row is False:
    row_added = False
    for p in range(len(itin)):
        if sum(sol_t[np.array(p_index_list)==p]) > Demand[p]:
            RMP.linear_constraints.add(
                lin_expr=[[list(sol_names_t[np.array(p_index_list)==p]), [1]*sum(np.array(p_index_list)==p)]],
                senses=['L'],
                rhs=[Demand[p]])
            sig_it_index_list.append(p)
            row_added = True
    if row_added is True:
        itcount_row += 1
        print("Added rows")
        RMP.solve()
        print("Cost            : {0:.5f}".format(RMP.solution.get_objective_value()))
        print("Solution status :", RMP.solution.get_status())
        sol_t = np.array(RMP.solution.get_values(np.arange(len(itin)).tolist()) + RMP.solution.get_values()[len(itin) + len(GA_labels) + len(flights) * len(actype_dfs.index):])
        sol_names_t = np.array(RMP.variables.get_names(np.arange(len(itin)).tolist()) + RMP.variables.get_names()[len(itin) + len(GA_labels) + len(flights) * len(actype_dfs.index):])
        obj = RMP.solution.get_objective_value()
        pi = np.array(RMP.solution.get_dual_values())
        sig_vect[sig_it_index_list] = np.array(RMP.solution.get_dual_values()[constr_pi_begin+len(flights):])
        Opt_Col = False
    else:
        Opt_Row = True

print("Total computational time is " + str(tm.time() - start_time) + " seconds")
print("Initial and final number of variables are respectively " + str(initial_num_var) + ' and ' + str(RMP.variables.get_num()))
print("Initial and final number of constraints are respectively " + str(initial_num_constr) + ' and ' + str(RMP.linear_constraints.get_num()))
print("Total number of columns added is " + str(len(vars_added)))
print("Total number of rows added is " + str(RMP.linear_constraints.get_num() - len(flights)-constr_pi_begin))
print("Total number of iterations done is " + str(itcount_col))
print("Total number of iterations through column while loop is " + str(itcount_col))
print("Total number of iterations through row while loop is " + str(itcount_row))

# RMP.write('rmp.lp')

solnames_f = np.array(RMP.variables.get_names()[len(itin):len(itin) + len(flights) * len(actype_dfs.index)])
sol_f = np.round(RMP.solution.get_values()[len(itin):len(itin) + len(flights) * len(actype_dfs.index)])
f_chosen = solnames_f[sol_f==1]
ix_A340 = [i for i, x in enumerate(f_chosen) if "A340" in x]
f_A340_chosen = f_chosen[ix_A340]

t_names_spilled = sol_names_t[sol_t>0]
t_spilled = sol_t[sol_t>0]

sol_index_list = [index for index in range(len(sol)) if sol[index] > 0.]
sol_names_list = sol_names[sol_index_list]