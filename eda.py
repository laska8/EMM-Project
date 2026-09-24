import json
import networkx as nx
import pandas as pd

with open("reply_networks/politics.json", "r") as fp:
    month_nets = json.load(fp)

results = []

for month_idx, month_net in enumerate(month_nets):

    G = nx.DiGraph()

    for user, replied_to in month_net.items():
        for target in replied_to:
            G.add_edge(user, target)

    results.append({
        "month": month_idx + 1,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "reciprocity": nx.reciprocity(G)
    })

df_months = pd.DataFrame(results)

print(df_months)