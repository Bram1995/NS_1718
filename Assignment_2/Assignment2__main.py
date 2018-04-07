import cplex
import numpy as np
import pandas as pd
import copy
from datetime import datetime,timedelta,date
from datetime import time as dt
import time as tm

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
bigM = 10000000
xl = pd.ExcelFile("Assignment2.xlsx")
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
actypes = actype_dfs.index[:4]
airport_list = sorted(list(set(flight_dfs['ORG'])))

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
#RMP.variables.add(obj=Fare,
#                  names=['t' + str(p) + '_x' for p in range(len(itin))])
for i in flights:
    for k in actype_dfs.index:
        RMP.variables.add(obj=[flight_dfs.loc[i,k]], names=['f_' + str(i) + '_' + str(k)], types='B')

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
        #  n+
        print('y_' + k + '_' + airport + '_' + str(time) + '_' + str(time2))
        #  n-
        print('y_' + k + '_' + airport + '_' + str(time1) + '_' + str(time))
        # f O +
        print('f_' + i + '_' + k for i in O_set.index)
        # f I -
        print('f_' + i + '_' + k for i in I_set.index)

        RMP.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=['y_' + k + '_' + airport + '_' + str(time) + '_' + str(time2)] + [
                'y_' + k + '_' + airport + '_' + str(time1) + '_' + str(time)] + ['f_' + i + '_' + k for i in O_set.index] + ['f_' + i + '_' + k for i in I_set.index],
                                       val=[-1, 1] + [-1] * len(O_set) + [1] * len(I_set))],
            senses=['E'],
            rhs=[0])

## add constraint set 9 total amount of AC constraint
for k in actypes:
    flight_set = time_check_dict[k]
    ga_set = time_check_ga_dict[k]
    rhs = int(actype_dfs.loc[k,'Units'])
    RMP.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=['f_' + i + '_' + k for i in flight_set.index] + [
            'y_' + k + '_' + ga_set.loc[j,'Airport'] + '_' + str(ga_set.loc[j,'Time1']) + '_' + str(ga_set.loc[j,'Time2'])
            for j in ga_set.index], val=[1] * len(flight_set) + [1] * len(ga_set))], senses=['L'], rhs=[rhs])


## add seperate constraints for busses -> get own capacity




## Add constraint set 4
# rhs_1 = []
# A_1 = np.zeros((len(flights), len(itin)))
# for i in range(len(flights)):
#     leg1_ind = dfs["Itinerary"].index[dfs["Itinerary"]["Leg 1"] == flights[i]].tolist()
#     leg2_ind = dfs["Itinerary"].index[dfs["Itinerary"]["Leg 2"] == flights[i]].tolist()
#     index_it = leg1_ind + leg2_ind
#     A_1[i, index_it] = 1
#     #cap = dfs['Flight'].loc[flights[i], 'Capacity']
#     dem1 = int(sum(dfs['Itinerary'].loc[dfs['Itinerary']['Leg 1'] == flights[i], 'Demand'].tolist()))
#     dem2 = int(sum(dfs['Itinerary'].loc[dfs['Itinerary']['Leg 2'] == flights[i], 'Demand'].tolist()))
#     rhs_1.append(float(dem1 + dem2)) #-cap
#     RMP.linear_constraints.add(
#         lin_expr=[[index_it, A_1[i, index_it].tolist()]],
#         senses=['G'],
#         rhs=[rhs_1[i]])


## Solve model
RMP.solve()
print("Solution status :", RMP.solution.get_status())
print("Cost            : {0:.5f}".format(RMP.solution.get_objective_value()))
print()
sol = np.array(RMP.solution.get_values())
sol_names = np.array(RMP.variables.get_names())
sol_index_list = [index for index in range(len(sol)) if sol[index] > 0.]
sol_names_list = sol_names[sol_index_list]




sol_names = np.array(RMP.variables.get_names())
obj = RMP.solution.get_objective_value()

# pi = np.array(RMP.solution.get_dual_values())
# sig_it_index_list = []  # list of indices of itineraries for which row constraint is added
# vars_added = []  # list of indices of recapture variables already added to model by column generation
# sig_vect = np.array([0]*len(dfs['Itinerary']))  # initially we do not have sigma -> set to 0 for pricing problem
RMP.write('rmp.lp')


##  COLUMN & ROW GENERATION
# Opt_Row = False
# Opt_Col = False
# p_index_list = np.arange(len(itin)).tolist()  #i row of p-numbers for all variables added
# while Opt_Row is False or Opt_Col is False:
#     # Column Generation loop
#     while Opt_Col is False:
#         col_added = False
#         RMP, p_index_list, col_added, vars_added = col_generation(RMP, dfs, pi, sig_vect, p_index_list, vars_added)
#         if col_added is True:
#             RMP.solve()
#             print("Cost            : {0:.5f}".format(RMP.solution.get_objective_value()))
#             sol = np.array(RMP.solution.get_values())
#             sol_names = np.array(RMP.variables.get_names())
#             obj = RMP.solution.get_objective_value()
#             pi = np.array(RMP.solution.get_dual_values()[:len(flights)])
#             if len(np.array(RMP.solution.get_dual_values()[len(flights):])) != 0:
#                 sig_vect[sig_it_index_list] = np.array(RMP.solution.get_dual_values()[len(flights):])
#             Opt_Row = False
#         else:  # No column added, so can move to row generation
#             Opt_Col = True
#
#     # Row Generation loop
#     while Opt_Row is False:
#         row_added = False
#         for p in range(len(itin)):
#             if sum(sol[np.array(p_index_list)==p]) > Demand[p]:
#                 RMP.linear_constraints.add(
#                     lin_expr=[[list(sol_names[np.array(p_index_list)==p]), [1]*sum(np.array(p_index_list)==p)]],
#                     senses=['L'],
#                     rhs=[Demand[p]])
#                 sig_it_index_list.append(p)
#                 row_added = True
#         if row_added is True:
#             RMP.solve()
#             print("Cost            : {0:.5f}".format(RMP.solution.get_objective_value()))
#             print("Solution status :", RMP.solution.get_status())
#             sol = np.array(RMP.solution.get_values())
#             sol_names = np.array(RMP.variables.get_names())
#             obj = RMP.solution.get_objective_value()
#             pi = np.array(RMP.solution.get_dual_values()[:len(flights)])
#             sig_vect[sig_it_index_list] = np.array(RMP.solution.get_dual_values()[len(flights):])
#             Opt_Col = False
#         else:
#             Opt_Row = True
#
# RMP.write('rmp.lp')


# ##  Get flow per flightleg between hubs and division of itnierary pax
# df_hubflights = dfs['Flight'].loc[dfs['Flight']["ORG"].isin(["AEP","EZE"]) & dfs['Flight']["DEST"].isin(["AEP","EZE"])]
# for i in iterrows(df_hubflights):
#     flow[i] = Q_i - sum_r t_ir + sum_r b_ri t_ri

