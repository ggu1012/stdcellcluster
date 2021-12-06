from torch_geometric.utils import from_networkx

import json
from networkx.readwrite import json_graph

fG = open('modfiles/graph.json')
G_nx = json.load(fG)

G = json_graph.node_link_graph(G_nx)

print(G)
data = from_networkx(G)
print(data)