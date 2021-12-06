import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import json
import matplotlib.pyplot as plt

fconn = open("modfiles/connectivity.json", 'r')
fgraph = open('modfiles/graph.json', 'w')
fmap = open('modfiles/id_map.json', 'w')
conn = json.load(fconn)

nodes_list = list(conn.keys())

G = nx.Graph()
G.add_nodes_from(nodes_list)

visited = dict()

for idx, cell in enumerate(conn):
    visited[cell] = idx

    edges_list = []
    io_type = []

    for neighbor in conn[cell]:
        io_type.append(int(neighbor[-1]))

    out_num = len(io_type) - sum(io_type)

    # 1: in, 0: out
    out_weight = 0 if out_num == 0 else 1 / out_num
    
    for neighbor in conn[cell]:
        splitted = neighbor.split('/')
        rm_neighbor = '/'.join(splitted[:-1])
        if rm_neighbor not in list(visited.keys()):
            G.add_edge(cell, rm_neighbor)

fmap.write(json.dumps(visited))

# for _node in G.nodes():
#     G.node[_node]['val'] = 0
#     G.node[_node]['test'] = 0


G_dict = json_graph.node_link_data(G)
fgraph.write(json.dumps(G_dict))

fmap.close()
fconn.close()
fgraph.close()

# pos = nx.spring_layout(G)
# nx.draw_networkx_nodes(G, pos)
# nx.draw_networkx_labels(G, pos)
# nx.draw_networkx_edges(G, pos)
# plt.savefig('xx.png')




