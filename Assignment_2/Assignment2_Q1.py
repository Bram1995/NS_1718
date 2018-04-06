import cplex
import numpy as np
import copy
import pandas as pd
from datetime import datetime,timedelta,time,date
import itertools

xl = pd.ExcelFile("Assignment2.xlsx")
dfs = {sheet: xl.parse(sheet) for sheet in xl.sheet_names}
bigM = 10000000
actype_dfs = dfs['Aircraft'].set_index(['Type'])
actype_dfs.at['Bus',['Units','Seats','TAT (min)']] = [1,216,0]



flight_dfs = dfs['Flight'].set_index(['Flight Number'])
flight_dfs.at[:,'Bus'] = bigM
indices = flight_dfs.loc[(flight_dfs['ORG'] == 'AEP') & (flight_dfs['DEST'] == 'EZE') |
                         (flight_dfs['ORG'] == 'EZE') & (flight_dfs['DEST'] == 'AEP')].index.tolist()
flight_dfs.at[indices, actype_dfs.index.values.tolist()] = 0
flight_dfs.at[indices, actype_dfs.index.values.tolist()[-1]] = 4500
flight_dfs = flight_dfs.fillna(bigM)
airport_list = sorted(list(set(flight_dfs['ORG'])))



nodes_dfs_dict = {}
for i in actype_dfs.index[:3]:
    tat = int(actype_dfs.loc[i,'TAT (min)'])
    flight_dataframe = copy.deepcopy(flight_dfs)
    flight_dataframe = flight_dataframe.loc[(flight_dataframe[i] != bigM),['ORG','DEST','Departure','Arrival',i]]
    flight_dataframe = flight_dataframe.loc[(flight_dataframe[i] != 0.),['ORG','DEST','Departure','Arrival',i]]
    flight_dataframe_sub_1 = flight_dataframe.loc[:,['ORG','Departure']].rename(columns ={'ORG':'Airport','Departure':'Time'})
    flight_dataframe_sub_2 = flight_dataframe.loc[:, ['DEST', 'Arrival']].rename(columns={'DEST': 'Airport', 'Arrival': 'Time'})
    for k in flight_dataframe_sub_2.index:
        time = datetime.combine(date(1,1,1),flight_dataframe_sub_2.loc[k,'Time'])
        flight_dataframe_sub_2.at[k,'Time'] = (time + timedelta(minutes= tat)).time()
    nodes_dataframe = copy.deepcopy(pd.concat([flight_dataframe_sub_1,flight_dataframe_sub_2])).sort_values(['Airport','Time'],ascending=[True,True])
    nodes_dataframe = nodes_dataframe.drop_duplicates(subset=['Airport','Time'])
    nodes_dfs_dict[i]= copy.deepcopy(nodes_dataframe)

ga_dict = {}
ga_list = []
for k in actype_dfs.index[:3]:
    for i in airport_list:
        ground_arc_dfs = pd.DataFrame(columns = ['Airport','Time1','Time2'])
        arc_selection = nodes_dfs_dict[k].loc[nodes_dfs_dict[k]['Airport']==i,:]
        for p,j in enumerate(arc_selection.index):
            time1 = arc_selection.iloc[p, 1]
            if p == len(arc_selection.index)-1:
                time2 =  arc_selection.iloc[0, 1]
                index = p+1
                ground_arc_dfs.at[index, ['Airport','Time1', 'Time2']] = [i,time1, time2]
            else:
                time2 = arc_selection.iloc[p+1, 1]
                index = p+1
                ground_arc_dfs.at[index, ['Airport', 'Time1', 'Time2']] = [i,time1, time2]
            ga_list.append(copy.deepcopy(index))
        ga_dict[k + '_' + i] = copy.deepcopy(ground_arc_dfs)



