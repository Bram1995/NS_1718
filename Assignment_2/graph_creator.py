def graph_creator(source, sink,total_nodes_73x):
    nodes_73x = total_nodes_73x.loc[(((source < sink) & (total_nodes_73x["Time"] >= source) & (total_nodes_73x["Time"] <= sink)) |
                                    ((source > sink) & ((total_nodes_73x["Time"] >= source) & (total_nodes_73x["Time"]<dt(23,59,59)) |
                                     ((total_nodes_73x["Time"] <= sink) & (total_nodes_73x["Time"] > dt(0, 0, 0))))))]

    flight_data1 = total_flight_links.loc[((source < sink) & (total_flight_links["Arrival"] > total_flight_links["Departure"]) &
                                          (total_flight_links["Departure"] >= source) &
                                          (total_flight_links["Arrival"]<= sink))]


    flight_data2 = total_flight_links.loc[((source > sink) & (total_flight_links["Arrival"] > total_flight_links["Departure"]) &
                                           (total_flight_links["Departure"] >= dt(0,0,0)) &
                                          (total_flight_links["Arrival"] <= sink)) |
                                          ((source > sink) &
                                        (total_flight_links["Departure"]>= source) &
                                        (total_flight_links["Arrival"]<= dt(23,59,59))) |
                                        (source > sink) & (total_flight_links["Departure"]>= source) &
                                        (total_flight_links["Arrival"] <= sink)]
    flight_data = pd.concat([flight_data1,flight_data2])

    arc_data = pd.DataFrame(columns=['ORG','DEST','Departure','Arrival','Label'])
    for k in np.unique(nodes_73x['Airport'].values):
        sub = nodes_73x.loc[(nodes_73x['Airport']==k)]
        for i in range(len(sub)-1):
            org = k
            dest = k
            departure = sub.iloc[i,1]
            arrival = sub.iloc[i+1,1]
            if departure < arrival:
                duration = datetime.combine(date.min, arrival) - datetime.combine(date.min, departure)
            if arrival < departure:
                duration = timedelta(1,0) - (datetime.combine(date.min, departure) - datetime.combine(date.min, arrival))
            if duration < timedelta(0,3*60*60):
                label = [duration.seconds*1000, 0, duration, 1]      #[total_time, number of duties, idle time]
                arc_data = arc_data.append(pd.DataFrame([[org,dest,departure,arrival,label]],columns=['ORG','DEST','Departure','Arrival','Label']),ignore_index=True)

    for i in flight_data.index:
        org = flight_data.loc[i, 'ORG']
        dest = flight_data.loc[i, 'DEST']
        departure = flight_data.loc[i, 'Departure']
        arrival = flight_data.loc[i, 'Arrival']
        if departure < arrival:
            duration = datetime.combine(date.min, arrival) - datetime.combine(date.min, departure)
        if arrival < departure:
            duration = timedelta(1,0) - (datetime.combine(date.min, departure) - datetime.combine(date.min, arrival))
        label = [duration.seconds, 1, timedelta(0), 0]  # [total_time, number of duties, idle time]
        arc_data = arc_data.append(pd.DataFrame([[org, dest, departure, arrival, label]],
                                               columns=['ORG', 'DEST', 'Departure', 'Arrival', 'Label']), ignore_index=True)

    ## Create graph:
    graph = {}
    for i,p in enumerate(nodes_73x.index):
        airport = nodes_73x.loc[p,'Airport']
        time = nodes_73x.loc[p,'Time']
        graph[airport + '_' + str(time)]={}
        arc_selection = arc_data.loc[(arc_data['ORG']==airport)&(arc_data['Departure']==time)]
        if arc_selection.empty == False:
            for j, t in enumerate(arc_selection.index):
                dest = arc_selection.at[t,'DEST']
                arr = arc_selection.at[t,'Arrival']
                label = arc_selection.at[t,'Label']
                graph[airport + '_' + str(time)][dest + '_' + str(arr)] = copy.deepcopy(label)
        else:
            del graph[airport + '_' + str(time)]
    graph['AEP_' + str(sink)] = {}  # add sink node so dijkstra can choose this one as last node
    return graph,flight_data,arc_data