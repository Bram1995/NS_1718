
import pandas as pd
#from col_generation import col_generation


## Load data
xl = pd.ExcelFile("Input_AE4424_Ass1P2.xlsx")
dfs = {sheet: xl.parse(sheet) for sheet in xl.sheet_names}
dfs['Flight'] = dfs['Flight'].set_index('Flight Number')



rhs_list = []
for i in dfs['Flight'].index.values:
    q = dfs['Flight'].loc[i,'Capacity']
    cap1 = int(sum(dfs['Itinerary'].loc[dfs['Itinerary']['Leg 1'] == i , 'Demand'].tolist()))
    cap2 = int(sum(dfs['Itinerary'].loc[dfs['Itinerary']['Leg 2'] == i , 'Demand'].tolist()))
    rhs_list.append(q-cap1-cap2)

print(rhs_list)

