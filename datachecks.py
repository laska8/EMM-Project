import json
import networkx as nx

with open("reply_networks/politics.json", "r") as fp:
    month_nets = json.load(fp)

month_1 = month_nets[0]

G = nx.DiGraph()

for user, replied_to in month_1.items():
    for target in replied_to:
        G.add_edge(user, target)

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Density:", nx.density(G))
print("Reciprocity:", nx.reciprocity(G))