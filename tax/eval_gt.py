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
import os
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from sklearn.metrics import f1_score, precision_score, accuracy_score, roc_auc_score


from transformers import LlamaForCausalLM, AutoTokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader, Dataset
from openai import OpenAI
LOGFILE='output.log'
# palm.configure(api_key=os.environ['PALM'])
# from transformers import LlamaForCausalLM, AutoTokenizer, LogitsProcessorList
# from torch.utils.data import DataLoader, Dataset
parser = argparse.ArgumentParser(description="Your script description")
# Add the configuration file argument
parser.add_argument("config_file", type=str, help="Path to the configuration file")
parser.add_argument("TOTAL", type=int, default=700, nargs="?", help="Number of total items to process")
parser.add_argument("--c", default=None,
                    help="Path to the checkpoint to resume training. Default is None.")
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
# openai_api_key = os.environ.get("OPENAI")
filename=f"{datapath}/parent_0shot_{TOTAL}.json"

# if not openai_api_key:
    # print("OpenAI API key not found in environment variables.")
# client = OpenAI(api_key=openai_api_key)

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

# model_path = "/scratch/yerong/.cache/pyllama/Llama-2-7b-hf/"
model_id="google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
logging.info(f'Yes id is : {tokenizer(["Yes"])}')
logging.info(f'No id is : {tokenizer(["No"])}')
device = "cuda:0" # You can set this to "cpu" if you don't have a GPU

# model = T5ForConditionalGeneration.from_pretrained(model_id, device_map="auto")
model = T5ForConditionalGeneration.from_pretrained(model_id).to(device)
checkpoint_to_resume = args.c
if checkpoint_to_resume:
    print(f'Loading checkpoint : {args.c}')
    logging.info(f'Loading checkpoint : {args.c}')
    model = PeftModel.from_pretrained(model, checkpoint_to_resume, is_trainable=True)
else:
    raise ValueError("No checkpoint specified")
# sentences = ["The house is wonderful.", "I like to work in NYC."]

# inputs = tokenizer([sentence for sentence in sentences], return_tensors="pt", padding=True)

# output_sequences = model.generate(
#     input_ids=inputs["input_ids"],
#     attention_mask=inputs["attention_mask"],
#     do_sample=False,  # disable sampling to test if batching affects output
# )

# model = LlamaForCausalLM.from_pretrained(
#   model_path,
#   torch_dtype=torch.float16,
#   device_map='auto',
# ).eval()
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# tokenizer.pad_token = "[PAD]"
# pad_token_id = tokenizer.encode(tokenizer.pad_token)[0]
# tokenizer.padding_side = "left"
device = "cuda:0" # You can set this to "cpu" if you don't have a GPU
# logging.info(f'Yes id is : {tokenizer(["Yes"])}')
# logging.info(f'No id is : {tokenizer(["No"])}')
# 11-27 02:16:11 INFO - query1.py:28 - Yes id is : {'input_ids': [[1, 3869]], 'attention_mask': [[1, 1]]}
# 11-27 02:16:11 INFO - query1.py:29 - No id is : {'input_ids': [[1, 1939]], 'attention_mask': [[1, 1]]}

def predict_next_token_batch(prompts, batch_size=10):
    predictions = []
    sentences = [item['prompt'] for item in prompts]
    # Split prompts into batches
    for i in tqdm.tqdm(range(0, len(sentences), batch_size), desc="Processing Batches", unit="batch"):
        batch_prompts = sentences[i:i + batch_size]
        # Tokenize prompts and convert to PyTorch tensors
        inputs = tokenizer([sentence for sentence in batch_prompts], return_tensors="pt", padding=True)

        output_sequences = model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            max_new_tokens=1,
            do_sample=False,  # disable sampling to test if batching affects output
        )

        batch_result=tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        predictions.extend(batch_result)
        # outputs = model(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"])
        # logits = outputs.logits[:, -1, :]
        # print(logits)

#         # Generate logits for the next token using the model
#         with torch.no_grad():
#             next_tokens = model.generate(
#     input_ids=input_ids["input_ids"],
#     attention_mask=input_ids["attention_mask"],
#     do_sample=False,  # disable sampling to test if batching affects output
#      output_scores=True, return_dict_in_generate=True
# )
#             logging.info(next_tokens)
#             print(len(next_tokens.scores))
#             print(len(next_tokens.scores[0].shape))
#             print(tokenizer.batch_decode(next_tokens, skip_special_tokens=True))
#             # outputs = model(input_ids=input_ids["input_ids"],
#     # attention_mask=input_ids["attention_mask"])
#             # logits = outputs.logits[:, -1, :]

#         # Process logits or do whatever you need with them
#         next_tokens_scores = logits  # Assuming logits_processor is not used in this function
#         next_tokens = torch.argmax(next_tokens_scores, dim=-1)
#         # Example: Extract probabilities for specific tokens (adjust token IDs as needed)
#         yes_prob = next_tokens_scores[:, 3869]
#         no_prob = next_tokens_scores[:, 1939]

#         # Calculate the difference in probabilities
#         prob_diff = yes_prob - no_prob

#         # Determine the predictions based on probability differences
#         batch_predictions = torch.where(prob_diff > 0, 1, -1).tolist()
    
#         # Append batch predictions to the overall predictions list
#         predictions.extend(batch_predictions)

    return predictions

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

for iteration, edge in tqdm.tqdm(enumerate(random.sample(list(core_graph.edges()), 10)), total=10):
# for iteration, edge in tqdm.tqdm(enumerate(core_graph.edges()), total=core_graph.number_of_edges()):
    parent_, kid_ = edge
    if parent_ == rootkey or kid_ == rootkey : continue
    total_edge_count += 1
    if len(list(core_graph.predecessors(parent_))) < 1 or \
    len(list(core_graph.predecessors(parent_))) == 1 and rootkey in list(core_graph.predecessors(parent_)):
        continue
    if len(list(core_graph.neighbors(parent_))) < 2: continue

    has_parent_count += 1

    hs = HASH(definitions[parent_]['summary']+definitions[kid_]['summary'])

    if hs in predictions: continue
    parent_label = get_first_label_without_n(definitions[parent_]['label'])
    kid_label = get_first_label_without_n(definitions[kid_]['label'])

    q_parent_label = f'"{parent_label}"'
    q_kid_label = f'"{kid_label}"'
    #POSTIVE

    

    prompt = '''Given two terms in a knowledge graph, your task is to determine whether they have a parent-child relationship (Yes/No).

    - Question:
"golf" represents the parent node term under consideration. 
 - "golf" : a game played on a large open course with 9 or 18 holes; the object is use as few strokes as possible in playing all the holes
"golf" is the subclass of "outdoor_game".
 - "outdoor_game" : an athletic game that is played outdoors
 Also "golf" has following existing childen: 
"professional_golf" : playing golf for money
"clock_golf" : a form of golf in which you putt from positions arranged on the circumference of a circle around the hole
"medal_play" : golf scoring by total strokes taken
Now we want to add "miniature_golf" as a new child to the term "golf"
 - "miniature_golf" : a novelty version of golf played with golf balls and putters on a miniature course featuring many obstacles
If we decide to add a new node "miniature_golf" as a child of "golf", it should conceptually become the consistent grandchild of"outdoor_game". Also "miniature_golf" is a sibling of "professional_golf", "clock_golf" and "medal_play" with a same granularity.

 Answer:
 Yes'''
    prompt+="\n\n    - Question:"
    prompt+= f"\n{q_parent_label} represents the parent node term under consideration. \n - {q_parent_label} : {definitions[parent_]['summary']}"
    # Get neighbors of the parent_ node
    predecessors_of_parent = list(core_graph.predecessors(parent_))

    # Filter out nodes that are equal to kid_
    filtered_predecessors = [predecessor for predecessor in predecessors_of_parent if predecessor != rootkey]
    filtered_predecessors = [predecessor for predecessor in filtered_predecessors if core_graph[predecessor][parent_]['weight'] == 1]

    # Take up to three random neighbors
    
    selected_predecessors = random.sample(filtered_predecessors, min(3, len(filtered_predecessors)))
    del filtered_predecessors
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
    neighbors_of_parent = list(core_graph.neighbors(parent_))

    # Filter out nodes that are equal to kid_
    filtered_neighbors = [neighbor for neighbor in neighbors_of_parent if neighbor != kid_]
    filtered_neighbors = [neighbor for neighbor in filtered_neighbors if core_graph[parent_][neighbor]['weight'] == 1]

    # Take up to three random neighbors
    selected_neighbors = random.sample(filtered_neighbors, min(3, len(filtered_neighbors)))
    del filtered_neighbors
    prompt+= f"\n Also {q_parent_label} has following existing childen: "
    # for k in selected_neighbors:
    #     node_definitions.add(k)

    try:
        for node in selected_neighbors:
            label = get_first_label_without_n(definitions[node]['label'])
            # logging.info(node)
            # logging.info(definitions[node])
            description = definitions[node]['summary']
            prompt += f"\n\"{label}\" : {description}"
    except:
        print('error')
        continue
    nei_labels = [get_first_label_without_n(definitions[node]['label']) for node in selected_neighbors]
    q_nei_labels = [f'"{label}"' for label in nei_labels]
    del nei_labels

    prompt+= f"\nNow we want to add {q_kid_label} as a new child to the term {q_parent_label}"
    prompt += f"\n - {q_kid_label} : {definitions[kid_]['summary']}"

    prompt+= f"\nIf we decide to add a new node {q_kid_label} as a child of {q_parent_label}, it should conceptually become the consistent grandchild of"
    if len(selected_predecessors) > 1:
        prompt+= f"{', '.join(q_pre_labels[:-1])} and {q_pre_labels[-1]}. "
    else:
        prompt+= f"{q_pre_labels[0]}. "

    prompt+= f"Also {q_kid_label} is a sibling of {', '.join(q_nei_labels[:-1])} and {q_nei_labels[-1]} with a same granularity."
    prompt+= f"\n\n Answer:\n"


    # prompt+= f'\n Question: Is {get_first_label_without_n(definitions[parent_]["label"])} a parent of {get_first_label_without_n(definitions[kid_]["label"])}?\n Answer:' 
    
    prompts.append({'prompt': prompt, 
        'label': core_graph[parent_][kid_]['weight'],
        'hs': hs,
        })

    if iteration <= 10:
        logging.info(prompt)


    del hs, kid_, kid_label, prompt, selected_predecessors



    # # NEGATIVE sample
    # if random.random() < 0.25:
    #     pass

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




batch_size = 20

# predict_palm_batch(prompts, batch_size)
# predict_llama_batch(prompts, batch_size)
# predict_gpt_batch(prompts)

# for prompt, output in zip(prompts, predictions):
#     logging.info(prompt['prompt'])
#     logging.info(output) 


predictions =predict_next_token_batch(prompts, batch_size)
# print(len(predictions))
# print(predictions)
# print(len(prompts))
predictions = [1 if entry == 'Yes' else -1 if entry == 'No' else -3 for entry in predictions]

result = [{'label': prompts[i]['label'], 'pred': predictions[i]} for i in range(len(prompts))]
# Filter out instances where the label is -3


labels = [entry['label']for entry in result]

valid_indices = [i for i, label in enumerate(labels) if label != -3]

filtered_labels = [labels[i] for i in valid_indices]
filtered_predictions = [predictions[i] for i in valid_indices]

# Calculate metrics
f1 = f1_score(filtered_labels, filtered_predictions, average='binary', pos_label=1)
precision = precision_score(filtered_labels, filtered_predictions, average='binary', pos_label=1)
accuracy = accuracy_score(filtered_labels, filtered_predictions)

# AUC score is calculated only if both classes are present
if len(set(filtered_labels)) == 2:
    auc_score = roc_auc_score(filtered_labels, filtered_predictions)
else:
    auc_score = None
print(args.c)
print(f"F1 Score: {f1:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Accuracy: {accuracy:.3f}")
if auc_score is not None:
    print(f"AUC Score: {auc_score:.3f}")
else:
    print("AUC Score: N/A")

logging.info(args.c)

logging.info(f"F1 Score: {f1:.3f}")
logging.info(f"Precision: {precision:.3f}")
logging.info(f"Accuracy: {accuracy:.3f}")
if auc_score is not None:
    logging.info(f"AUC Score: {auc_score:.3f}")
else:
    logging.info("AUC Score: N/A")
# output_sequences = model.generate(**inputs, max_new_tokens=20, do_sample=True, top_p=0.9)
#         weight = core_graph[parent][kid]['weight']
#         if weight == -1:
#             print(parent, kid)
#     except:
#         logging.info(definitions[parent])
#         logging.info(definitions[kid])
#         logging.info(core_graph[parent][kid])





