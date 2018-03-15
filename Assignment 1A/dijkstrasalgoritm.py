import numpy as np

def graph_creator(dfs):
    # graph=dict()
    origins = dfs['Arcs'].From
    destinations = dfs['Arcs'].To
    node_array = np.append(origins.values, destinations.values)

    destinations_list = []
    for i in range(len(origins)):
        sub_destination_list = []
        sub_costs_list = []
        sub_dict = dict()
        node = origins[i]
        dfs['Arcs'].Cost
        arcs_selection = dfs['Arcs'].loc[dfs['Arcs'].From == node]
        costs_selection = arcs_selection.Cost.values
        neighbours_selection = arcs_selection.To.values
        for j in range(len(neighbours_selection)):
            sub_destination_list.append(neighbours_selection[j])
            sub_costs_list.append(costs_selection[j])
        sub_dict = dict(zip(sub_destination_list, sub_costs_list))
        destinations_list.append(sub_dict)
    graph = dict(zip(node_array, destinations_list))
    return graph





# --------------------------------------------------------------------------------------------------------------------

def dijkstra(graph, src, dest, visited=None, distances=None, predecessors=None):
    # source of code of this function: http://www.gilles-bertrand.com/2014/03/dijkstra-algorithm-python-example-source-code-shortest-path.html
    """ calculates a shortest path tree routed in src
    """
    if visited is None:
        visited = []
    if distances is None:
        distances = {}
    if predecessors is None:
        predecessors = {}
    # a few sanity checks
    if src not in graph:
        raise TypeError('The root of the shortest path tree cannot be found')
    if dest not in graph:
        raise TypeError('The target of the shortest path cannot be found')
        # ending condition
    if src == dest:
        # We build the shortest path and display it
        path = []
        pred = dest
        while pred != None:
            path.append(pred)
            pred = predecessors.get(pred, None)
        path_cost = distances[dest]
        # print('shortest path: ' + str(path) + " cost=" + str(distances[dest]))
        path=np.flipud(path)
        return path, path_cost
    else:
        # if it is the initial  run, initializes the cost

        if not visited:
            distances[src] = 0
        # visit the neighbors
        for neighbor in graph[src]:
            if neighbor not in visited:
                new_distance = distances[src] + graph[src][neighbor]
                if new_distance < distances.get(neighbor, float('inf')):
                    distances[neighbor] = new_distance
                    predecessors[neighbor] = src
        # mark as visited
        visited.append(src)
        # now that all neighbors have been visited: recurse
        # select the non visited node with lowest distance 'x'
        # run Dijskstra with src='x'

        unvisited = {}
        for k in graph:
            if k not in visited:
                unvisited[k] = distances.get(k, float('inf'))
        x = min(unvisited, key=unvisited.get)
        # print(x)
        return dijkstra(graph, x, dest, visited, distances, predecessors)






