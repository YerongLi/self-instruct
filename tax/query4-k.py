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
prefix = '''Your task is to assess the consistent feasibility of adding a new node term as a child to a designated parent node, considering the parent of the parent of the specified parenting node.

    - Question: 
"employment" represents the parent node term under consideration. 
 - "employment" : the occupation for which you are paid
"employment" is the subclass of "occupation".
 - "occupation" : the principal activity in your life that you do to earn money
Now we want to add "gambling" as a new child to the term "employment".
 - "gambling" : the act of playing for stakes in the hope of winning (including the payment of a price for a chance to win a prize)
If we decide to add a new node "gambling" as a child of "employment", it should conceptually become the consistent grandchild of "occupation".

 Answer:
No

 Explanation:
Based on the given definitions, "gambling" does not fit the concept of "employment" as a subclass of "occupation". "Gambling" is not a type of occupation or a principal activity in one's life that is done to earn money. It is a form of entertainment or leisure activity. Therefore, adding "gambling" as a child of "employment" would not be consistent with the concept of "occupation" and would not make sense in this hierarchy.

    - Question: 
"skiing" represents the parent node term under consideration. 
 - "skiing" : a sport in which participants must travel on skis
"skiing" is the subclass of "sport".
 - "sport" : an active diversion requiring physical exertion and competition
Now we want to add "cross-country_skiing" as a new child to the term "skiing".
 - "cross-country_skiing" : the sport of skiing across the countryside (rather than downhill)
If we decide to add a new node "cross-country_skiing" as a child of "skiing", it should conceptually become the consistent grandchild of "sport".

 Answer:
Yes

 Explanation:
Based on the given information, it is consistent to add "cross-country_skiing" as a child of "skiing" and have it be the grandchild of "sport". This is because "skiing" is already defined as a subclass of "sport", and "cross-country_skiing" is a specific type of skiing that falls under the broader category of "sport". Therefore, adding "cross-country_skiing" as a child of "skiing" would not contradict the existing relationship between "skiing" and "sport".\n''' 
# for iteration, edge in tqdm.tqdm(enumerate(random.sample(list(core_graph.edges()), 10)), total=10):
iter_count = 0 
for iteration, edge in tqdm.tqdm(enumerate(core_graph.edges()), total=core_graph.number_of_edges()):
    parent_, kid_ = edge
    if parent_ == rootkey or kid_ == rootkey : continue
    total_edge_count += 1
    if len(list(core_graph.predecessors(parent_))) < 1 or \
    len(list(core_graph.predecessors(parent_))) == 1 and rootkey in list(core_graph.predecessors(parent_)):
        continue
    has_parent_count += 1

    hs = HASH(definitions[parent_]['summary']+definitions[kid_]['summary'])

    if hs in predictions: continue
    iter_count+= 1
    parent_label = get_first_label_without_n(definitions[parent_]['label'])
    kid_label = get_first_label_without_n(definitions[kid_]['label'])

    q_parent_label = f'"{parent_label}"'
    q_kid_label = f'"{kid_label}"'
    
    #POSTIVE

    prompt = prefix
    

    prompt+= f"\n    - Question:\n{q_parent_label} represents the parent node term under consideration. \n - {q_parent_label} : {definitions[parent_]['summary']}"
    # Get neighbors of the parent_ node
    predecessors_of_parent = list(core_graph.predecessors(parent_))

    # Filter out nodes that are equal to kid_
    filtered_predecessors = [predecessor for predecessor in predecessors_of_parent if predecessor != rootkey]

    # Take up to three random neighbors
    selected_predecessors = random.sample(filtered_predecessors, min(3, len(filtered_predecessors)))
    pre_labels = [get_first_label_without_n(definitions[node]['label']) for node in selected_predecessors]
    q_pre_labels = [f'"{label}"' for label in pre_labels]
    del pre_labels
    
    if len(selected_predecessors) > 1:
    
        prompt+= f"\n{q_parent_label} is the subclass of {', '.join(q_pre_labels[:-1])} and {q_pre_labels[-1]}."
    else:
        prompt+= f"\n{q_parent_label} is the subclass of {q_pre_labels[0]}."

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
    prompt+= f"\nNow we want to add {q_kid_label} as a new child to the term {q_parent_label}."
    prompt += f"\n - {q_kid_label} : {definitions[kid_]['summary']}"

    prompt+= f"\nIf we decide to add a new node {q_kid_label} as a child of {q_parent_label}, it should conceptually become the consistent grandchild of "
    if len(selected_predecessors) > 1:
        prompt+= f"{', '.join(q_pre_labels[:-1])} and {q_pre_labels[-1]}."
    else:
        prompt+= f"{q_pre_labels[0]}."

    prompt+= f"\n\n Answer:\n"


    # prompt+= f'\n Question: Is {get_first_label_without_n(definitions[parent_]["label"])} a parent of {get_first_label_without_n(definitions[kid_]["label"])}?\n Answer:' 
    
    prompts.append({'prompt': prompt, 
        'label': core_graph[parent_][kid_]['weight'],
        'hs': hs,
        })

    logging.info(prompt)

    if iter_count <= 10:
        logging.info(prompt)


    del hs, kid_, kid_label, prompt, selected_predecessors



    # NEGATIVE sample
    if random.random() < (1.5 if '0shot' in filename else 0.2):
        parents_of_parent = set(core_graph.predecessors(parent_))

        # Find grandparents (parents of parents of parent_)
        grandparents_of_parent = set([grandparent for parent in parents_of_parent for grandparent in core_graph.predecessors(parent)])

        # Find the set of children of grandparents that are not in parents_of_parent
        children_of_grandparents_not_in_parents_of_parent = set()
        for grandparent in grandparents_of_parent:
            children_of_grandparent = set(core_graph.successors(grandparent))
            children_of_grandparents_not_in_parents_of_parent.update(children_of_grandparent - parents_of_parent)

        # Randomly sample one element and name it grand_
        if children_of_grandparents_not_in_parents_of_parent:
            grand_ = random.sample(children_of_grandparents_not_in_parents_of_parent, 1).pop()
        else:
            continue
        grand_label = get_first_label_without_n(definitions[grand_]['label'])
        q_grand_label = f'"{grand_label}"'

        hs = HASH(definitions[parent_]['summary']+definitions[grand_]['summary'])

        prompt = prefix
    

        prompt+= f"\n    - Question:\n{q_parent_label} represents the parent node term under consideration. \n - {q_parent_label} : {definitions[parent_]['summary']}"
    # Get neighbors of the parent_ node
        predecessors_of_parent = list(core_graph.predecessors(parent_))

        # Filter out nodes that are equal to kid_
        filtered_predecessors = [predecessor for predecessor in predecessors_of_parent if predecessor != rootkey]

        # Take up to three random neighbors
        selected_predecessors = random.sample(filtered_predecessors, min(3, len(filtered_predecessors)))
        pre_labels = [get_first_label_without_n(definitions[node]['label']) for node in selected_predecessors]
        q_pre_labels = [f'"{label}"' for label in pre_labels]
        del pre_labels
        
        if len(selected_predecessors) > 1:
        
            prompt+= f"\n{q_parent_label} is the subclass of {', '.join(q_pre_labels[:-1])} and {q_pre_labels[-1]}."
        else:
            prompt+= f"\n{q_parent_label} is the subclass of {q_pre_labels[0]}."

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
        prompt+= f"\nNow we want to add {q_kid_label} as a new child to the term {q_parent_label}."
        prompt += f"\n - {q_grand_label} : {definitions[grand_]['summary']}"

        prompt+= f"\nIf we decide to add a new node {q_grand_label} as a child of {q_parent_label}, it should conceptually become the consistent grandchild of "
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

        if iteration <= 10:
            logging.info(prompt)

    if min_pair is None or edge_list_len < min_len:
        min_pair = (parent_, kid_)
        min_len = edge_list_len

    if max_pair is None or edge_list_len > max_len:
        max_pair = (parent_, kid_)
        max_len = edge_list_len
    # Check if we need to sample additional negative pairs
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


