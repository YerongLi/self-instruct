import logging
import os
import pickle
import random
import tqdm
import networkx as nx
import torch
from transformers import LlamaForCausalLM, AutoTokenizer
TOTAL = 300
logging.basicConfig(
    format='%(asctime)s %(levelname)-4s - %(filename)-6s:%(lineno)d - %(message)s',
    level=logging.INFO,
    filename='./output.log',
    datefmt='%m-%d %H:%M:%S')

logging.info(f'Logger start: {os.uname()[1]}')

model_path = "/scratch/yerong/.cache/pyllama/Llama-2-7b-hf/"

model = LlamaForCausalLM.from_pretrained(
  model_path,
  torch_dtype=torch.float16,
  device_map='auto',
).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)
device = "cuda:0" # You can set this to "cpu" if you don't have a GPU
def predict_next_token(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate the next token using the model
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]

    # Extract probabilities for "Yes" and "No"
    print(logits.shape)
    # Extract the probability for "Yes"
    # print(logits[tokenizer.convert_tokens_to_ids(["Yes"])])
    yes_prob = logits[0][tokenizer.convert_tokens_to_ids(["Yes"])].item()

    # Calculate the probability for "No"
    no_prob = logits[0][tokenizer.convert_tokens_to_ids(["No"])].item()

    # Calculate the difference in probabilities
    prob_diff = yes_prob - no_prob

    # Determine the prediction based on probability difference
    if prob_diff > 0:
        prediction = 1
    else:
        prediction = -1

    return  prediction


prompt = "The quick brown fox jumps over the lazy dog. "
next_token = predict_next_token(prompt)
print("Predicted Token:", next_token)

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
for edge in tqdm.tqdm(core_graph.edges()):
    parent_, kid_ = edge
    neighbors = list(core_graph.neighbors(parent_))
    neighbors = random.sample(neighbors, min(6, len(neighbors)))
    edge_list = edges_within_k_edges(core_graph, parent_, kid_)
    # Sample 6 edges from edge_list
    sampled_edges = random.sample(edge_list, min(6, len(edge_list)))

    # Get all nodes from sampled edges
    nodes = set()
    for edge in sampled_edges:
        nodes.add(edge[0])
        nodes.add(edge[1])
    node_definitions = nodes.copy()
    for n in neighbors: node_definitions.add(n)

    # Sample all negative pairs within nodes
    negative_pairs = []
    for node1 in nodes:
        for node2 in nodes:
            if node1 != node2 and (node1, node2) not in edge_list and (node2, node1) not in edge_list:
                negative_pairs.append((node1, node2))
    random.shuffle(negative_pairs)
    # Sample additional negative pairs from core_graph.nodes()
    num_additional_pairs = 6 - (len(nodes) * (len(nodes) - 1) - len(sampled_edges))

    additional_pairs = []
    while len(additional_pairs) < num_additional_pairs:
        node1 = random.choice(list(core_graph.nodes()))
        node2 = random.choice(list(nodes))
        node_definitions.add(node2)
        if (node1, node2) not in edge_list and (node1, node2) not in sampled_edges:
            additional_pairs.append((node1, node2))



    # Create the prompt
    prompt = "Given two terms in a knowledge graph, your task is to determine whether they have a parent-child relationship at the same granularity as the example pairs."
    node_definitions.add(parent_)
    node_definitions.add(kid_)
    # random.shuffle(node_definitions)
    for node in node_definitions:
        label = get_first_label_without_n(definitions[node]['label'])
        # logging.info(node)
        # logging.info(definitions[node])
        description = definitions[node]['summary']
        prompt += f"Definitions: {label} : {description}\n"

    prompt += "\n"
    pairs = [(get_first_label_without_n(definitions[parent_]['label']), get_first_label_without_n(definitions[kid]['label']), 'Yes') for kid in neighbors if kid is not kid_]
    for edge in sampled_edges:
        parent = edge[0]
        kid = edge[1]
        if parent == parent_ and kid == kid_: continue
        parent_label = get_first_label_without_n(definitions[parent]['label'])
        kid_label = get_first_label_without_n(definitions[kid]['label'])
        pairs.append((parent_label, kid_label, 'Yes'))

    # Combine negative pairs and additional pairs
    negative_pairs += additional_pairs

    # Create the negative samples prompt
    # prompt += "\nNegative Samples:\n\n"
    for pair in (negative_pairs )[:6]:
        parent = pair[0]
        kid = pair[1]
        if parent == parent_ and kid == kid_: continue
        parent_label = get_first_label_without_n(definitions[parent]['label'])
        kid_label = get_first_label_without_n(definitions[kid]['label'])
        pairs.append((parent_label, kid_label, 'No'))
    random.shuffle(pairs)

    for pair in pairs:
        prompt+= f'\n Question: Is {pair[0]} a parent of {pair[1]}?\n Answer: {pair[2]}' 
    prompt+= f'\n Question: Is {get_first_label_without_n(definitions[parent_]["label"])} a parent of {get_first_label_without_n(definitions[kid_]["label"])}?\n Answer:' 
    
    logging.info(prompt)
    logging.info(predict_next_token(prompt))

    edge_list_len = len(edge_list)

    if min_pair is None or edge_list_len < min_len:
        min_pair = (parent, kid)
        min_len = edge_list_len

    if max_pair is None or edge_list_len > max_len:
        max_pair = (parent, kid)
        max_len = edge_list_len
    # Check if we need to sample additional negative pairs
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
    logging.info(f"The first 100 neighbors of {definitions[parent]['label']} are:")
    # for neighbor in list(core_graph.neighbors(parent))[:100]:
        # logging.info(definitions[neighbor])

logging.info(f"The minimum length of the edge lists is {min_len}.")
logging.info(f"The maximum length of the edge lists is {max_len}.")
logging.info([])
logging.info(core_graph)
#     try:
#         weight = core_graph[parent][kid]['weight']
#         if weight == -1:
#             print(parent, kid)
#     except:
#         logging.info(definitions[parent])
#         logging.info(definitions[kid])
#         logging.info(core_graph[parent][kid])
