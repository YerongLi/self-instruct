'''
Your task is to assess the consistent feasibility of adding a new node term as a child to a designated parent node, considering the parent of the parent of the specified parenting node.

<P> represents the parent node term under consideration.
<P> : <Description>
<A>, <B>, and <C> are parents of <P>.
<A> : <Description>
<B> : <Description>
<C> : <Description>
If we choose to introduce a new node <X> as a child of <P>, it should conceptually become the consistent grandchild of A, B, and C.
<X> : <Description>
'''

'''
Your task involves determining the consistent addition of a new node term as a child to a parenting node, considering the parent of that parent of the parenting node.

Consider <P> as the parent node term. The structure is as follows:

<A>, <B>, and <C> serve as parents of <P>.
<A>: <Description>
<B>: <Description>
<C>: <Description>
'''

'''
Your task is to assess the feasibility of consistently introducing a new node term as a child of a designated parent node, taking into account the immediate parent and the grandparent of the specified parenting node.
Consider the parent node term as <P> with the following structure:

<P> : <Description>
<A>, <B>, and <C> serve as the parents of <P>.
<A> : <Description>
<B> : <Description>
<C> : <Description>
If we choose to incorporate a new node <X> as a child of <P>, it is imperative that <X> maintains a consistent conceptual relationship as a grandchild of nodes A, B, and C.
'''

# 12-01 20:59:46 INFO - query3.py:367 - Given multiple child terms associated with a parent term in a knowledge graph, your task is to evaluate the possi
# bility of introducing a provided candidate term as a new child under the same parent. The new term should align with the existing children, forming sib
# lings at the same hierarchical level. Please provide a thorough and detailed explanation for your decision, taking into account the relationships withi
# n the knowledge graph.

#  Question: 
# "entity" is the parenting node. 
# "entity" : that which is perceived or known or inferred to have its own distinct existence (living or nonliving)"entity" has following existing childen
# : 
# "physical_entity" : an entity that has physical existence
# Now we want to add "abstraction" as a new child to the term "entity"
# "abstraction" : a general concept formed by extracting common features from specific examples
# With the information that "physical_entity" is a child node of "entity". We can add "abstraction" as a child node of "entity" without any conflicts. As
#  a result, "abstraction" is a sibling of "physical_entity" with a same granularity.

#  Answer:
# Yes

#  Explanation:



import argparse
import json
import time
import logging
import hashlib
import os
import pickle
import random
import shutil
import tqdm
# import networkx as nx
import torch
import requests
import json
import openai
import google.generativeai as palm
import os

from transformers import LlamaForCausalLM, AutoTokenizer, LogitsProcessorList
from torch.utils.data import DataLoader, Dataset
from openai import OpenAI
LOGFILE='output.log'
palm.configure(api_key=os.environ['PALM'])
# from transformers import LlamaForCausalLM, AutoTokenizer, LogitsProcessorList
# from torch.utils.data import DataLoader, Dataset
parser = argparse.ArgumentParser(description="Your script description")
# Add the configuration file argument
parser.add_argument("config_file", type=str, help="Path to the configuration file")
parser.add_argument("TOTAL", type=int, default=700, nargs="?", help="Number of total items to process")
parser.add_argument("mode", type=str, default='llama', nargs="?", help="Prediction mode")

def HASH(input_string):
    # Use SHA-256 for deterministic hashing
    hash_object = hashlib.sha256(input_string.encode())
    hash_value = int.from_bytes(hash_object.digest(), byteorder='big')

    return str(hash_value)
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
openai_api_key = os.environ.get("OPENAI")
filename=f"{datapath}/parent_kshot_{TOTAL}.json"

if not openai_api_key:
    print("OpenAI API key not found in environment variables.")
client = OpenAI(api_key=openai_api_key)

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
    filename=LOGFILE,
    datefmt='%m-%d %H:%M:%S')

logging.info(f'Logger start: {os.uname()[1]}')

model_path = "/scratch/yerong/.cache/pyllama/Llama-2-7b-hf/"

model = None
# model = LlamaForCausalLM.from_pretrained(
#   model_path,
#   torch_dtype=torch.float16,
#   device_map='auto',
# ).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = "[PAD]"
pad_token_id = tokenizer.encode(tokenizer.pad_token)[0]
tokenizer.padding_side = "left"
device = "cuda:0" # You can set this to "cpu" if you don't have a GPU
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

prompts = []
predictions = {}
if os.path.exists(filename):
    with open(filename, "r") as f:
        predictions = json.load(f)
total_edge_count = 0
has_parent_count = 0


# START
'''
Your task is to assess the consistent feasibility of adding a new node term as a child to a designated parent node, considering the parent of the parent of the specified parenting node.

<P> represents the parent node term under consideration.
<P> : <Description>
<A>, <B>, and <C> are parents of <P>.
<A> : <Description>
<B> : <Description>
<C> : <Description>
If we choose to introduce a new node <X> as a child of <P>, it should conceptually become the consistent grandchild of A, B, and C.
<X> : <Description>
'''
prefix = '''You have a term within a taxonomy graph along with its known children. Your goal is to add this term as a new child to a parent term. Your task is to assess whether, upon adding this term as a child to the parent term, the existing children of this term will consistently become grandchildren of the newly designated parent term. Please provide a detailed explanation for your evaluation.
    
    - Question:
"blood_sport" represents the child node term under consideration. 
 - "blood_sport" : sport that involves killing animals (especially hunting)
"blood_sport" has children of "hunt", "bullfighting" and "cockfighting".
 - "hunt" : the pursuit and killing or capture of wild animals regarded as a sport
 - "bullfighting" : the activity at a bullfight
 - "cockfighting" : participation in the sport of matching gamecocks in a cockfight
Now we want to add "blood_sport" as a new child to the term "sport".
 - "sport" : an active diversion requiring physical exertion and competition
If we decide to add a new node "blood_sport" as a child of "sport", it should conceptually become the consistent grandchild of "hunt", "bullfighting" a
nd "cockfighting".

 Answer:
Yes
 Explanation:
hunt: Originally a child of "blood_sport," it will now be a grandchild of "sport" in the expanded hierarchy. This is because "blood_sport" is becoming a child of "sport," making "hunt" a consistent grandchild.

bullfighting: Similar to "hunt," "bullfighting" was a direct child of "blood_sport" and will now be a consistent grandchild of "sport" after the addition of "blood_sport" as a child of "sport."

cockfighting: In the same manner, "cockfighting" was a child of "blood_sport" and will become a consistent grandchild of "sport" once "blood_sport" is added as a child of "sport."

In summary, the addition of "blood_sport" as a child to the term "sport" ensures that its existing children ("hunt," "bullfighting," and "cockfighting") consistently become grandchildren of the newly designated parent term "sport."
'''
iter_count = 0
neg_count = 0
predictions = {}
for iteration, edge in tqdm.tqdm(enumerate(random.sample(list(core_graph.edges()), 10)), total=10):

# for iteration, edge in tqdm.tqdm(enumerate(core_graph.edges()), total=core_graph.number_of_edges()):
    parent_, kid_ = edge
    if parent_ == rootkey or kid_ == rootkey : continue
    total_edge_count += 1
    if len(list(core_graph.predecessors(parent_))) < 1 or \
    len(list(core_graph.predecessors(parent_))) == 1 and rootkey in list(core_graph.predecessors(parent_)):
        continue
    has_parent_count += 1

    hs = HASH(definitions[parent_]['summary']+definitions[kid_]['summary'])

    # if hs in predictions: continue
    iter_count+= 1
    parent_label = get_first_label_without_n(definitions[parent_]['label'])
    kid_label = get_first_label_without_n(definitions[kid_]['label'])

    q_parent_label = f'"{parent_label}"'
    q_kid_label = f'"{kid_label}"'
    # logging.info('q_kid_label')
    # logging.info(q_kid_label)
    #POSTIVE

    prompt = prefix
    

    prompt+= f"\n    - Question:\n{q_kid_label} represents the child node term under consideration. \n - {q_kid_label} : {definitions[kid_]['summary']}"
    # Get neighbors of the parent_ node
    children_of_kid = list(core_graph.neighbors(kid_))

    # Filter out nodes that are equal to kid_
    filtered_grands = [grand for grand in children_of_kid if grand != rootkey]

    # Take up to three random neighbors
    selected_grands = random.sample(filtered_grands, min(3, len(filtered_grands)))
    g_labels = [get_first_label_without_n(definitions[node]['label']) for node in selected_grands]
    q_g_labels = [f'"{label}"' for label in g_labels]
    del g_labels
    if len(selected_grands) == 0 : continue
    if len(selected_grands) > 1:
    
        prompt+= f"\n{q_kid_label} has children of {', '.join(q_g_labels[:-1])} and {q_g_labels[-1]}."
    else:
        prompt+= f"\n{q_kid_label} has one children of {q_g_labels[0]}."

    # for k in selected_predecessors:
    #     node_definitions.add(k)

    try:
        for node in selected_grands:
            label = get_first_label_without_n(definitions[node]['label'])
            # logging.info(node)
            # logging.info(definitions[node])
            description = definitions[node]['summary']
            prompt += f"\n - \"{label}\" : {description}"
    except:
        print('error')
        continue
    prompt+= f"\nNow we want to add {q_kid_label} as a new child to the term {q_parent_label}."
    prompt += f"\n - {q_parent_label} : {definitions[parent_]['summary']}"

    prompt+= f"\nIf we decide to add a new node {q_kid_label} as a child of {q_parent_label}, it should conceptually become the consistent grandchild of "
    if len(selected_grands) > 1:
        prompt+= f"{', '.join(q_g_labels[:-1])} and {q_g_labels[-1]}."
    else:
        prompt+= f"{q_g_labels[0]}."

    prompt+= f"\n\n Answer:\n"


    # prompt+= f'\n Question: Is {get_first_label_without_n(definitions[parent_]["label"])} a parent of {get_first_label_without_n(definitions[kid_]["label"])}?\n Answer:' 
    
    prompts.append({'prompt': prompt, 
        'label': core_graph[parent_][kid_]['weight'],
        'hs': hs,
        })

    logging.info(prompt)

    if iter_count <= 10:
        logging.info(prompt)

    p_prompt = prompt
    del hs, kid_label, q_kid_label,prompt, filtered_grands, selected_grands



    # NEGATIVE sample
    continue
    if random.random() < (1.5 if '0shot' in filename else 0.2):
    # if True:
        # Assuming 'core_graph' is your graph structure and 'parent_' is a specific node

        # Get the parents of the parent node
        parents_of_parent = set(core_graph.predecessors(parent_))

        # Initialize an empty set to store the siblings
        siblings = set()

        # Iterate over each parent of the parent node
        for parent_of_parent in parents_of_parent:
            # Add all successors (children) of the parent of the parent node to the siblings set
            siblings.update(core_graph.successors(parent_of_parent))

        # Remove the original parent node from the siblings set
        siblings.discard(parent_)

        # Initialize an empty set to store the parents of the siblings
        parents_of_siblings = set()

        # Iterate over each sibling
        for sibling in siblings:
            # Add all parents of the sibling node to the set
            parents_of_siblings.update(core_graph.predecessors(sibling))

        # Now, 'parents_of_siblings' set contains all the parents of the siblings

        parents_of_kid = set(core_graph.predecessors(kid_))

        # Initialize an empty set to store the parents of parents (grandparents) of the kid_
        parents_of_parents_of_kid = set()

        # Iterate over each parent of the kid_
        for parent_of_kid in parents_of_kid:
            # Add all parents of the parent of the kid_ to the set
            parents_of_parents_of_kid.update(core_graph.predecessors(parent_of_kid))




        difference_set = parents_of_siblings - parents_of_parents_of_kid

        # Check if the difference set is not empty
        if difference_set:
            # Randomly sample one node from the difference set as f_grandparent
            f_grandparent = random.sample(difference_set, 1)[0]

            # Get the neighbors of f_grandparent from core_graph
            f_grandparent_neighbors = set(core_graph.neighbors(f_grandparent))

            # Check if f_grandparent has neighbors
            if f_grandparent_neighbors:
                # Randomly pick one f_parent from the neighbors of f_grandparent
                f_parent_ = random.sample(f_grandparent_neighbors, 1)[0]
            else:
                continue
        else:
            continue
        # DEBUG
        del parent_

        # Randomly sample one element and name it grand_

        f_parent_label = get_first_label_without_n(definitions[f_parent_]['label'])
        q_f_parent_label = f'"{f_parent_label}"'
        del f_parent_label

        hs = HASH(definitions[f_parent_]['summary']+definitions[kid_]['summary'])

        prompt = prefix
    

        prompt+= f"\n    - Question:\n{q_f_parent_label} represents the parent node term under consideration. \n - {q_f_parent_label} : {definitions[f_parent_]['summary']}"
    # Get neighbors of the parent_ node
        # Filter out nodes that are equal to kid_

        # Take up to three random neighbors
        predecessors_of_parent = list(core_graph.predecessors(f_parent_))

        # Filter out nodes that are equal to kid_
        filtered_predecessors = [predecessor for predecessor in predecessors_of_parent if predecessor != rootkey]
        selected_predecessors = random.sample(filtered_predecessors, min(3, len(filtered_predecessors)))
        selected_predecessors = set(selected_predecessors)
        selected_predecessors.add(f_grandparent)
        assert f_grandparent in core_graph.predecessors(f_parent_), 'f_grandparent in core_graph.predecessors(f_parent_)'
        assert f_grandparent not in parents_of_parents_of_kid, 'f_grandparent not in parents_of_parents_of_kid' # DEBUG
        pre_labels = [get_first_label_without_n(definitions[node]['label']) for node in selected_predecessors]
        q_pre_labels = [f'"{label}"' for label in pre_labels]
        del pre_labels
        
        if len(selected_predecessors) ==  0: continue
        if len(selected_predecessors) > 1:
        
            prompt+= f"\n{q_f_parent_label} is the subclass of {', '.join(q_pre_labels[:-1])} and {q_pre_labels[-1]}."
        else:
            prompt+= f"\n{q_f_parent_label} is the subclass of {q_pre_labels[0]}."

            # for k in selected_predecessors:
            #     node_definitions.add(k)

        try:
            for node in selected_predecessors:
                label = get_first_label_without_n(definitions[node]['label'])
                # logging.info(node)
                # logging.info(definitions[node])
                description = definitions[node]['summary']
                prompt += f"\n - \"{label}\" : {description}"
        except:
            print('error')
            continue
        prompt+= f"\nNow we want to add {q_kid_label} as a new child to the term {q_f_parent_label}."
        prompt += f"\n - {q_kid_label} : {definitions[kid_]['summary']}"

        prompt+= f"\nIf we decide to add a new node {q_kid_label} as a child of {q_f_parent_label}, it should conceptually become the consistent grandchild of "
        if len(selected_predecessors) > 1:
            prompt+= f"{', '.join(q_pre_labels[:-1])} and {q_pre_labels[-1]}."
        else:
            prompt+= f"{q_pre_labels[0]}."

        prompt+= f"\n\n Answer:\n"


        # prompt+= f'\n Question: Is {get_first_label_without_n(definitions[parent_]["label"])} a parent of {get_first_label_without_n(definitions[kid_]["label"])}?\n Answer:' 
        
        prompts.append({'prompt': prompt, 
            'label': -1,
            'hs': hs,
            })
        neg_count+= 1
        # if iter_count <= 10:
        logging.info('Negative')
        logging.info(prompt)
        logging.info(definitions[f_grandparent])
        logging.info('===========================================================')
        logging.info(p_prompt)

    if min_pair is None or edge_list_len < min_len:
        min_pair = (parent_, kid_)
        min_len = edge_list_len

    if max_pair is None or edge_list_len > max_len:
        max_pair = (parent_, kid_)
        max_len = edge_list_len
    # Check if we need to sample additional negative pairs
print(f'Number of negative samples is {neg_count}')
logging.info(f"has_parent_count:{has_parent_count}")
logging.info(f"total_edge_count: {total_edge_count}")

def save_predictions_to_file(predictions):
    with open(filename, "w") as file:
        json.dump(predictions, file, indent=4)  # Add 'indent' parameter for pretty formatting
    print(f"Predictions saved to {filename} === Total {len(predictions)}")


# PALM
def predict_palm_batch(prompts, batch_size=10):
    predictions = {}

    # Check if the predictions file exists
    if os.path.exists(filename):
        with open(filename, "r") as f:
            predictions = json.load(f)
    try:
        for item in prompts:
            # Check if the sentence is already in the predictions
            if item['hs'] in predictions: continue
            sentence = item['prompt']
            result = palm.generate_text(prompt=sentence).result
            predictions[item['hs']] = {'i' : sentence, 'o': result}
    except KeyboardInterrupt as e:
        print(f"Interupt")
        save_predictions_to_file(predictions)
    except Exception as e:
        print(f"An error occurred: {e}")
        save_predictions_to_file(predictions)
        return
    save_predictions_to_file(predictions)

# LlaMA
def predict_llama_batch(prompts, batch_size=10):
    # Check if the predictions file exists
    predictions = {}
    if os.path.exists(filename):
        backup_filename = filename + ".backup"
        shutil.copyfile(filename, backup_filename)
        print(f"Backup created: {backup_filename}")
        with open(filename, "r") as f:
            predictions = json.load(f)
    prompts = [item for item in prompts if item['hs'] not in predictions]
    # Split prompts into batches
    print(f'Total Number of Queries are {len(prompts)}')

    try:
        for z in tqdm.tqdm(range(0, len(prompts), batch_size), desc="Processing Batches", unit="batch"):
            batch_prompts = prompts[z:z + batch_size]
            batch_sentences = [item['prompt'] for item in batch_prompts]
            # Tokenize prompts and convert to PyTorch tensors
            i_ids = tokenizer(batch_sentences, return_tensors="pt", padding=True).to(device)

            # Generate logits for the next token using the model


            sentence_lengths = []

            # Assuming input_ids is a torch tensor
            # Iterate through each batch
            for batch in i_ids:
                # Iterate through each sentence in the batch
                for sentence in i_ids['input_ids']:
                    # Get the length of the sentence
                    sentence_length = torch.sum(sentence != 0)  # Assuming 0 is the padding token
                    sentence_lengths.append(sentence_length.item())
            c_ids, outputs = [], []
            with torch.no_grad():
                o_ids = model.generate(**i_ids, max_new_tokens=200, num_beams=10,
    num_return_sequences=1,no_repeat_ngram_size=1,)
                for i in range(len(batch_prompts)):
                    o_id = o_ids[i][o_ids[i] != pad_token_id][sentence_lengths[i]:]
                    c_ids.append(o_id)
                # print(len(o_ids))
                # print(o_ids)
                
                # print(c_ids)
                for i in range(len(batch_prompts)):
                    outputs.append(tokenizer.decode(c_ids[i], skip_special_tokens=True))

                    predictions[batch_prompts[i]['hs']] = {'i' : batch_sentences[i], 'o' : outputs[i]}
    except KeyboardInterrupt as e:
        print(f"Interupt")
        save_predictions_to_file(predictions)
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     save_predictions_to_file(predictions)
    #     return
    save_predictions_to_file(predictions)

def predict_llama_batch(prompts, batch_size=10):
    # Check if the predictions file exists
    predictions = {}
    if os.path.exists(filename):
        backup_filename = filename + ".backup"
        shutil.copyfile(filename, backup_filename)
        print(f"Backup created: {backup_filename}")
        with open(filename, "r") as f:
            predictions = json.load(f)
    prompts = [item for item in prompts if item['hs'] not in predictions]
    # Split prompts into batches
    print(f'Total Number of Queries are {len(prompts)}')

    try:
        for z in tqdm.tqdm(range(0, len(prompts), batch_size), desc="Processing Batches", unit="batch"):
            batch_prompts = prompts[z:z + batch_size]
            batch_sentences = [item['prompt'] for item in batch_prompts]
            # Tokenize prompts and convert to PyTorch tensors
            i_ids = tokenizer(batch_sentences, return_tensors="pt", padding=True).to(device)

            # Generate logits for the next token using the model


            sentence_lengths = []

            # Assuming input_ids is a torch tensor
            # Iterate through each batch
            for batch in i_ids:
                # Iterate through each sentence in the batch
                for sentence in i_ids['input_ids']:
                    # Get the length of the sentence
                    sentence_length = torch.sum(sentence != 0)  # Assuming 0 is the padding token
                    sentence_lengths.append(sentence_length.item())
            c_ids, outputs = [], []
            with torch.no_grad():
                o_ids = model.generate(**i_ids, max_new_tokens=200, num_beams=10,
    num_return_sequences=1,no_repeat_ngram_size=1,)
                for i in range(len(batch_prompts)):
                    o_id = o_ids[i][o_ids[i] != pad_token_id][sentence_lengths[i]:]
                    c_ids.append(o_id)
                # print(len(o_ids))
                # print(o_ids)
                
                # print(c_ids)
                for i in range(len(batch_prompts)):
                    outputs.append(tokenizer.decode(c_ids[i], skip_special_tokens=True))

                    predictions[batch_prompts[i]['hs']] = {'i' : batch_sentences[i], 'o' : outputs[i]}
    except KeyboardInterrupt as e:
        print(f"Interupt")
        save_predictions_to_file(predictions)
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     save_predictions_to_file(predictions)
    #     return
    save_predictions_to_file(predictions)

def predict_gpt_batch(prompts, batch_size=20):
    # Check if the predictions file exists
    predictions = {}
    if os.path.exists(filename):
        backup_filename = filename + ".backup"
        shutil.copyfile(filename, backup_filename)
        print(f"Backup created: {backup_filename}")
        with open(filename, "r") as f:
            predictions = json.load(f)
    const_prompts = [item for item in prompts if item['hs'] not in predictions]
    del prompts
    # url = "https://api.openai.com/v1/completions"
    # headers = {
    #     "Content-Type": "application/json",
    #     "Authorization": f"Bearer {openai_api_key}"
    # }

    try:
        for z in tqdm.tqdm(range(0, len(const_prompts), batch_size), desc="Processing Batches", unit="batch"):
            batch_prompts = const_prompts[z:z + batch_size]
            responses = client.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=[p['prompt'] for p in batch_prompts],
                temperature=0,
                max_tokens=512,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            # time.sleep(16)


            # Access individual responses in the list
            for i in range(len(batch_prompts)):
                predictions[batch_prompts[i]['hs']] = {'i' : batch_prompts[i]['prompt'], 'o': responses.choices[i].text}
            save_predictions_to_file(predictions)
        

    except KeyboardInterrupt as e:
        print(f"Interupt")
        save_predictions_to_file(predictions)
    except Exception as e:
        print(e)
        save_predictions_to_file(predictions)
    save_predictions_to_file(predictions)


batch_size = 4

# predict_palm_batch(prompts, batch_size)
# predict_llama_batch(prompts, batch_size)
# predict_gpt_batch(prompts)

# for prompt, output in zip(prompts, predictions):
#     logging.info(prompt['prompt'])
#     logging.info(output) 


