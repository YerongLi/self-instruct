import logging
import os
import pickle
import random
import tqdm
# import networkx as nx
import torch
from transformers import LlamaForCausalLM, AutoTokenizer, LogitsProcessorList
TOTAL = 300
logging.basicConfig(
    format='%(asctime)s %(levelname)-4s - %(filename)-6s:%(lineno)d - %(message)s',
    level=logging.INFO,
    filename='./2output.log',
    datefmt='%m-%d %H:%M:%S')

logging.info(f'Logger start: {os.uname()[1]}')

model_path = "/scratch/yerong/.cache/pyllama/Llama-2-7b-hf/"


def predict_next_token(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate the next token using the model
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]

    # Extract probabilities for "Yes" and "No"
    # Extract the probability for "Yes"
    # print(logits[tokenizer.convert_tokens_to_ids(["Yes"])])
    next_tokens_scores = logits_processor(input_ids, logits)
    next_tokens = torch.argmax(next_tokens_scores, dim=-1)
    # print(next_tokens_scores.shape)
    # Calculate the probability for "No"
    yes_prob = next_tokens_scores[0][3869].item()

    no_prob = next_tokens_scores[0][1939].item()
    # logging.info(f'{yes_prob}    {no_prob}')
    # Calculate the difference in probabilities
    prob_diff = yes_prob - no_prob

    # Determine the prediction based on probability difference
    if prob_diff > 0:
        prediction = 1
    else:
        prediction = -1
    # logging.info('next_token')
    # logging.info(next_tokens)
    return prediction
    # return  tokenizer.decode(next_tokens)


def get_first_label_without_n(label_str):
    # Split the label string by "||"
    labels = label_str.split('||')

    # Find the first label that contains ".n."
    for label in labels:
        if not ".n." in label:
            return label

    # If no label contains ".n.", return the first label
    print(label[0])
    return labels[0]
def edges_within_k_edges(graph, parent, child, k=3):
    # Create a set to store the visited nodes
    visited = set()

    # Create a list to store the edges within k edge distances
    ans = []

    # Recursive function to perform DFS
    def dfs(node, depth):
        # Mark the node as visited
        visited.add(node)

        # Check if the depth is greater than or equal to k
        if depth >= k:
            return

        # Iterate over the predecessors of the node
        for neighbor in graph.predecessors(node):
            if neighbor not in visited:
                # Add the edge to the list of edges within k edge distances
                ans.append((neighbor, node))
                dfs(neighbor, depth + 1)

        # Iterate over the neighbors of the node
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                # Add the edge to the list of edges within k edge distances
                ans.append((node, neighbor))
                dfs(neighbor, depth + 1)

    dfs(parent, 0)
    dfs(child, 0)

    # Return the list of edges within k edge distances
    ans = [(parent, kid) for parent, kid in ans if parent != rootkey and kid != rootkey]
    return ans


# Load the definitions variable from the file
with open('../../TaxoComplete/core_graph.pkl', 'rb') as f:
    core_graph = pickle.load(f)
with open('../../TaxoComplete/definitions.pkl', 'rb') as f:
    definitions = pickle.load(f)
rootkey = None

for key, value in definitions.items():
    if value['label'].strip() == '' and value['summary'].strip() == '':
        print(f"Key: {key}, Value: {value}")
        rootkey = key
        break
ans = -0x7f7f7f7f
single_neighbor_count = 0
zero_neighbor_count = 0
multiple_neighbor_count = 0

# print(definitions)
max_node = None
for node in core_graph.nodes():
    if node == rootkey: continue
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
    if node == rootkey: continue
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

min_pair = None
max_pair = None
result = []



if min_pair is not None:
    parent, kid = min_pair
    logging.info("Minimum pair:")
    logging.info(definitions[parent])
    logging.info(definitions[kid])

if max_pair is not None:
    parent, kid = max_pair
    logging.info("Maximum pair:")
    logging.info(definitions[parent])
    logging.info(definitions[kid])

logging.info(f"The minimum length of the edge lists is {min_len}.")
logging.info(f"The maximum length of the edge lists is {max_len}.")
logging.info(core_graph)
#     try:
#         weight = core_graph[parent][kid]['weight']
#         if weight == -1:
#             print(parent, kid)
#     except:
#         logging.info(definitions[parent])
#         logging.info(definitions[kid])
#         logging.info(core_graph[parent][kid])

