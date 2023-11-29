import argparse
import json
import logging
import os
import pickle
import random
import tqdm
# import networkx as nx

parser = argparse.ArgumentParser(description="Your script description")
# Add the configuration file argument
parser.add_argument("config_file", type=str, help="Path to the configuration file")
parser.add_argument("TOTAL", type=int, default=700, nargs="?", help="Number of total items to process")

args = parser.parse_args()
TOTAL = args.TOTAL

config_file = args.config_file
# Read the configuration file
with open(config_file) as f:
    config = json.load(f)

# Get the configuration values
# Extract the base filename without the ".taxo" extension
datapath = config['taxofilename'].split('/')[:-1]
datapath = '/'.join(datapath)
print(datapath)

print(TOTAL)
print(TOTAL)
print(TOTAL)
print(TOTAL)
print(TOTAL)
print(TOTAL)
logging.basicConfig(
    format='%(asctime)s %(levelname)-4s - %(filename)-6s:%(lineno)d - %(message)s',
    level=logging.INFO,
    filename=f'output.log',
    datefmt='%m-%d %H:%M:%S')

logging.info(f'Logger start: {os.uname()[1]}')

model_path = "/scratch/yerong/.cache/pyllama/Llama-2-7b-hf/"


# logging.info(f'Yes id is : {tokenizer(["Yes"])}')
# logging.info(f'No id is : {tokenizer(["No"])}')
# 11-27 02:16:11 INFO - query1.py:28 - Yes id is : {'input_ids': [[1, 3869]], 'attention_mask': [[1, 1]]}
# 11-27 02:16:11 INFO - query1.py:29 - No id is : {'input_ids': [[1, 1939]], 'attention_mask': [[1, 1]]}


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
with open(f'{datapath}/core_graph_{TOTAL}.pkl', 'rb') as f:
    core_graph = pickle.load(f)
with open(f'{datapath}/definitions.pkl', 'rb') as f:
    definitions = pickle.load(f)
rootkey = None

# Count the number of edges where a parent has more than one child
count = 0
non_root_edges = 0

for parent, child in core_graph.edges():
    if parent == rootkey:
        continue

    if len(list(core_graph.neighbors(parent))) > 1:
        count += 1

    non_root_edges += 1

# Calculate the percentage of edges where a parent has more than one child
percentage = (count / non_root_edges) * 100

logging.info(f"Total number of edges: {core_graph.number_of_edges()}")
logging.info(f"Number of edges (excluding root): {non_root_edges}")
logging.info(f"Number of edges where a parent has more than one child: {count}")
logging.info(f"Percentage of edges where a parent has more than one child: {percentage:.2f}%")

for key, value in definitions.items():
    try:
        if value['label'].strip() == '' and value['summary'].strip() == '':
            print(f"Key: {key}, Value: {value}")
            rootkey = key
            break
    except:
        continue
single_neighbor_count = 0
zero_neighbor_count = 0
multiple_neighbor_count = 0
multiple_neighbor_nodes, multiple_neighbor_nodes_6 = [], []
# print(definitions)
max_node = None


ans = -0x7f7f7f7f
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

    # Store nodes with multiple neighbors
    if length > 6:
        multiple_neighbor_nodes_6.append(node)
    else:
        multiple_neighbor_nodes.append(node)
# print(definitions[max_node])
# print(core_graph.neighbors(max_node))
logging.info(f"Max number of the neighbors are {ans}")
logging.info(f"Number of nodes with zero neighbors: {zero_neighbor_count}")
logging.info(f"Number of nodes with one neighbor: {single_neighbor_count}")
logging.info(f"Number of nodes with two or more neighbors: {multiple_neighbor_count}")
# for node in multiple_neighbor_nodes_6:
#     logging.info(definitions[node])
#     logging.info(len([_ for _ in core_graph.neighbors(node)]))
# logging.info("Nodes with 2-6 neighbors:")
# for node in multiple_neighbor_nodes:
#     neighbors = list(core_graph.neighbors(node))
#     logging.info(definitions[node])
#     for nei in neighbors:
#         logging.info(f"  ***     {definitions[nei]}")
# logging.info("====   ====")
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
count_edges = 0
for iteration, edge in tqdm.tqdm(enumerate(core_graph.edges()), total=core_graph.number_of_edges()):
    parent_, kid_ = edge
    if parent_ == rootkey or kid_ == rootkey : continue
    count_edges+= 1
    edge_list = edges_within_k_edges(core_graph, parent_, kid_)

    edge_list_len = len(edge_list)

    if min_pair is None or edge_list_len < min_len:
        min_pair = (parent_, kid_)
        min_len = edge_list_len

    if max_pair is None or edge_list_len > max_len:
        max_pair = (parent_, kid_)
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

logging.info(f"The minimum length of the edge lists is {min_len}.")
logging.info(f"The maximum length of the edge lists is {max_len}.")
logging.info(core_graph)
logging.info(f"Number of edges : {count_edges}")


or iteration, edge in tqdm.tqdm(enumerate(list(core_graph.edges())[:11]), total=11):
# for iteration, edge in tqdm.tqdm(enumerate(core_graph.edges()), total=core_graph.number_of_edges()):
    parent_, kid_ = edge
    if len(core_graph.neighbors(parent_)) < 2:
        continue
    if parent_ == rootkey or kid_ == rootkey : continue
    
    prompt = "Given two terms in a knowledge graph, your task is to determine whether they have a parent-child relationship with examples"
    node_definitions.add(parent_)
    node_definitions.add(kid_)
    # random.shuffle(node_definitions)
    try:
        for node in node_definitions:
            label = get_first_label_without_n(definitions[node]['label'])
            # logging.info(node)
            # logging.info(definitions[node])
            description = definitions[node]['summary']
            prompt += f"{label} : {description}\n"
    except:
        continue
    parent_label = get_first_label_without_n(definitions[parent_]['label'])
    kid_label = get_first_label_without_n(definitions[kid_]['label'])
    pairs = []
    pairs.append((parent_label, kid_label, 'Yes'))


    # # Combine negative pairs and additional pairs
    # negative_pairs += additional_pairs

    # # Create the negative samples prompt
    # # prompt += "\nNegative Samples:\n\n"
    # for pair in (negative_pairs )[:6]:
    #     parent = pair[0]
    #     kid = pair[1]
    #     if parent == parent_ and kid == kid_: continue
    #     parent_label = get_first_label_without_n(definitions[parent]['label'])
    #     kid_label = get_first_label_without_n(definitions[kid]['label'])
    #     pairs.append((parent_label, kid_label, 'No'))
    # random.shuffle(pairs)

    for pair in pairs:
        prompt+= f'\n Question: Is {pair[0]} a parent of {pair[1]}?\n Answer: {pair[2]}' 
    prompt+= f'\n Explanation: '
    # prompt+= f'\n Question: Is {get_first_label_without_n(definitions[parent_]["label"])} a parent of {get_first_label_without_n(definitions[kid_]["label"])}?\n Answer:' 
    
    prompts.append({'prompt': prompt, 'label': core_graph[parent_][kid_]['weight']})
    # predicted_label = predict_next_token(prompt)
    if iteration <= 10:
        logging.info(prompt)
        # logging.info(predicted_label)
    if core_graph[parent_][kid_]['weight'] == -1: count_neg_label+= 1
    # result.append((parent_, kid_, {'label': core_graph[parent_][kid_]['weight'], 'pred' : predicted_label}))
    edge_list_len = len(edge_list)

    if min_pair is None or edge_list_len < min_len:
        min_pair = (parent_, kid_)
        min_len = edge_list_len

    if max_pair is None or edge_list_len > max_len:
        max_pair = (parent_, kid_)
        max_len = edge_list_len
    # Check if we need to sample additional negative pairs

def predict_batch(prompts, batch_size=10):
    predictions = []
    sentences = [item['prompt'] for item in prompts]
    # Split prompts into batches
    for i in tqdm.tqdm(range(0, len(sentences), batch_size), desc="Processing Batches", unit="batch"):
        batch_prompts = sentences[i:i + batch_size]
        # Tokenize prompts and convert to PyTorch tensors
        input_ids = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)

        output_sequences = model.generate(**inputs, do_sample=True, top_p=0.9)
        outputs = tokenizer.batch_decode(output_sequences, skip_special_tokens=True))

        predictions.extend(outputs)

    return predictions
batch_size = 8
predictions = predict_batch(prompts)

for prompt, output in zip(prompts, predictions):
    logging.info(prompt['prompt'])
    logging.info(prediction) 


