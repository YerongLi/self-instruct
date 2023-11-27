import logging
import os
import pickle
import random
import tqdm
import networkx as nx

TOTAL = 300
logging.basicConfig(
    format='%(asctime)s %(levelname)-4s - %(filename)-6s:%(lineno)d - %(message)s',
    level=logging.INFO,
    filename='./output.log',
    datefmt='%m-%d %H:%M:%S')

logging.info(f'Logger start: {os.uname()[1]}')

def edges_within_k_edges(graph, parent, child, k=2):
    # Create a set to store the visited nodes
    visited = set()

    # Create a list to store the edges within k edge distances
    ans = []

    # Perform DFS on the parent node
    for node in [x for x in graph.neighbors(parent)] + [x for x in graph.predecessors(parent)]:
        # Mark the node as visited
        visited.add(node)

        # Check if the node is within k edge distances of the child node
        if nx.has_path(graph, node, child) and nx.shortest_path_length(graph, node, child) <= k:
            # Iterate over the neighbors of the node
            for neighbor in graph.neighbors(node):
                # Check if the neighbor is not visited and is not the parent node
                if neighbor not in visited and neighbor != parent:
                    # Add the edge to the list of edges within k edge distances
                    ans.append((node, neighbor))

    # Perform DFS on the child node
    for node in [x for x in graph.neighbors(child)] + [x for x in graph.predecessors(child)]:
        # Mark the node as visited
        visited.add(node)

        # Check if the node is within k edge distances of the parent node
        if nx.has_path(graph, node, parent) and nx.shortest_path_length(graph, node, parent) <= k:
            # Iterate over the neighbors of the node
            for neighbor in graph.neighbors(node):
                # Check if the neighbor is not visited and is not the child node
                if neighbor not in visited and neighbor != child:
                    # Add the edge to the list of edges within k edge distances
                    ans.append((node, neighbor))

    return ans


# Load the definitions variable from the file
with open('../../TaxoComplete/core_graph.pkl', 'rb') as f:
    core_graph = pickle.load(f)
with open('../../TaxoComplete/definitions.pkl', 'rb') as f:
    definitions = pickle.load(f)

ans = -0x7f7f7f7f
single_neighbor_count = 0
zero_neighbor_count = 0
multiple_neighbor_count = 0

# print(definitions)
max_node = None
for node in core_graph.nodes():
    if definitions[node]['label'] == ' ': continue
    if not core_graph.has_node(node): continue

    length = len([_ for _ in core_graph.neighbors(node)])

    if length == 0:
        zero_neighbor_count += 1
    elif length == 1:
        single_neighbor_count += 1
    else:
        multiple_neighbor_count += 1

    if length > ans:
        ans = length
        max_node = node

# print(definitions[max_node])
# print(core_graph.neighbors(max_node))
logging.info(f"Max number of the neighbors are {ans}")
logging.info(f"Number of nodes with zero neighbors: {zero_neighbor_count}")
logging.info(f"Number of nodes with one neighbor: {single_neighbor_count}")
logging.info(f"Number of nodes with two or more neighbors: {multiple_neighbor_count}")
logging.info("====")
ans = -0x7f7f7f7f
single_neighbor_count = 0
zero_neighbor_count = 0
multiple_neighbor_count = 0

# print(definitions)
max_node = None
for node in core_graph.nodes():
    if definitions[node]['label'] == ' ': continue
    if not core_graph.has_node(node): continue

    length = len([_ for _ in core_graph.predecessors(node)])

    if length == 0:
        zero_neighbor_count += 1
    elif length == 1:
        single_neighbor_count += 1
    else:
        multiple_neighbor_count += 1

    if length > ans:
        ans = length
        max_node = node

# print(definitions[max_node])
# print(core_graph.neighbors(max_node))
logging.info(f"Max number of the predecessors are {ans}")
logging.info(f"Number of nodes with zero predecessors: {zero_neighbor_count}")
logging.info(f"Number of nodes with one predecessor: {single_neighbor_count}")
logging.info(f"Number of nodes with two or more predecessors: {multiple_neighbor_count}")

min_len = float('inf')
max_len = float('-inf')
for edge in tqdm.tqdm(core_graph.edges()):
    parent, kid = edge
    edge_list = edges_within_k_edges(core_graph, parent, kid)
    edge_list_len = len(edge_list)
    min_len = min(min_len, edge_list_len)
    max_len = max(max_len, edge_list_len)

# Print the minimum and maximum length of the edge lists
print(f"The minimum length of the edge lists is {min_len}.")
print(f"The maximum length of the edge lists is {max_len}.")
#     try:
#         weight = core_graph[parent][kid]['weight']
#         if weight == -1:
#             print(parent, kid)
#     except:
#         logging.info(definitions[parent])
#         logging.info(definitions[kid])
#         logging.info(core_graph[parent][kid])
