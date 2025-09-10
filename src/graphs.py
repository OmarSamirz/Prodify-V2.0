import pandas as pd
from pyvis.network import Network
import networkx as nx
from constants import GPC_PATH

# create graph
G = nx.DiGraph()

df = pd.read_excel(GPC_PATH).drop_duplicates(subset=["BrickTitle"])

# add edges from DataFrame
for _, row in df.iterrows():
    G.add_edge(row["BrickTitle"], row["ClassTitle"])
    G.add_edge(row["ClassTitle"], row["FamilyTitle"])
    G.add_edge(row["FamilyTitle"], row["SegmentTitle"])

# create PyVis network
net = Network(height="1200px", width="100%", directed=True, notebook=False)
net.from_nx(G)

# style nodes
for node in net.nodes:
    if node["id"] in df["BrickTitle"].values:
        node["color"] = "lightgreen"
        node["shape"] = "box"
    elif node["id"] in df["ClassTitle"].values:
        node["color"] = "lightblue"
    elif node["id"] in df["FamilyTitle"].values:
        node["color"] = "orange"
    elif node["id"] in df["SegmentTitle"].values:
        node["color"] = "red"

# make nodes static (non-draggable)
net.set_options("""
{
  "physics": {
    "enabled": true
  },
  "interaction": {
    "dragNodes": false
  }
}
""")

# save as interactive HTML
net.show("gpc_hierarchy.html", notebook=False)