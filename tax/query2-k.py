#     "Given two terms in a knowledge graph, your task is to determine whether they have a parent-child relationship and given a very detailed explanatio
# n on your decision.\n Question: \ncausal_agent : any entity that produces an effect or is responsible for events or results\nphysical_entity : an entity that has physical existence\n Is physical_entity a parent of causal_agent?\n Answer: Yes\n Explanation: ": "Causality is a property of an entity that
#  produces an effect or is responsible for events or results. Anything that produces an effect or is responsible for events or results must have physica
# l existence.",
#     "Given two terms in a knowledge graph, your task is to determine whether they have a parent-child relationship and given a very detailed explanatio
# n on your decision.\n Question: \nmatter : that which has mass and occupies space\nphysical_entity : an entity that has physical existence\n Is physica
# l_entity a parent of matter?\n Answer: Yes\n Explanation: ": "A physical entity is something that has physical existence. Matter is that which has mass and occupies space. Therefore, a physical entity is a parent of matter.",

import argparse
import json
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
import time
from openai import OpenAI
import google.generativeai as palm
import os

from transformers import LlamaForCausalLM, AutoTokenizer, LogitsProcessorList
from torch.utils.data import DataLoader, Dataset
openai_api_key = os.environ.get("OPENAI")

if not openai_api_key:
    print("OpenAI API key not found in environment variables.")
client = OpenAI(api_key=openai_api_key)
LOGFILE='output.log'
palm.configure(api_key=os.environ['PALM'])
# from transformers import LlamaForCausalLM, AutoTokenizer, LogitsProcessorList
# from torch.utils.data import DataLoader, Dataset
parser = argparse.ArgumentParser(description="Your script description")
# Add the configuration file argument
parser.add_argument("config_file", type=str, help="Path to the configuration file")
parser.add_argument("TOTAL", type=int, default=700, nargs="?", help="Number of total items to process")
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

print(datapath)
filename=f"{datapath}/prediction_kshot_{TOTAL}.json"

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

# model = None
# model = LlamaForCausalLM.from_pretrained(
#   model_path,
#   torch_dtype=torch.float16,
#   device_map='auto',
#   low_cpu_mem_usage=True,
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
# with open(f'{datapath}/predictions_1shot_{TOTAL}.json', "r") as f:
#     exemplars = json.load(f)
prefix = '''Given two terms in a knowledge graph, your task is to determine whether they have a parent-child relationship and given a very detailed explanation on your decision.
 
    - Question: 
"peeper" : an animal that makes short high-pitched sounds
"animal" : a living organism characterized by voluntary movement
 Is "animal" a parent of "peeper"?
 Answer:
Yes
 Explanation:
"Animal" is a general category that includes various living organisms with voluntary movement. "Peeper" falls under the category of "animal" based on its definition, indicating that it is a specific type of animal.
The hierarchical relationship is akin to a parent-child relationship, where "animal" serves as the parent category, and "peeper" is a specific type or child category within that broader classification.

    - Question: 
weightlift : bodybuilding by exercise that involves lifting weights
incline_bench_press : a bench press performed on an inclined bench
 Is weightlift a parent of incline_bench_press?
  Answer:
No
  Explanation
"Weightlift" is a general term that refers to bodybuilding exercises involving lifting weights. This can include various exercises such as weightlifting, dumbbell exercises, and more.

"Incline Bench Press" is a specific type of exercise that involves performing a bench press on an inclined bench. While it does involve lifting weights, it is a distinct and specific exercise within the broader category of weightlifting.

Given this analysis, "Weightlift" is a broader category encompassing a variety of weightlifting exercises, not specifically inclined bench presses. "Incline Bench Press" is a specific type of weightlifting exercise, but it does not represent the entirety of bodybuilding exercises involving lifting weights.

Therefore, in this knowledge graph, "weightlift" is not a parent of "incline_bench_press." Instead, "weightlift" is a broader category that includes various types of weightlifting exercises, and "incline_bench_press" is a specific exercise falling within the weightlifting domain. The relationship is better described as a broader category (weightlift) and a specific type or instance (incline_bench_press), rather than a parent-child relationship.'''
# filename=f"{datapath}/predictions_kshot_{TOTAL}.json"
predictions = {}
if os.path.exists(filename):
    with open(filename, "r") as f:
        predictions = json.load(f)
predictions = {}
for iteration, edge in tqdm.tqdm(enumerate(random.sample(list(core_graph.edges()), 20)), total=20):
# for iteration, edge in tqdm.tqdm(enumerate(core_graph.edges()), total=core_graph.number_of_edges()):
    parent_, kid_ = edge
    if len(list(core_graph.neighbors(parent_))) < 2:
        continue
    if parent_ == rootkey or kid_ == rootkey : continue
    hs = HASH(definitions[parent_]['summary']+definitions[kid_]['summary'])
    if hs in predictions: continue

    parent_label = get_first_label_without_n(definitions[parent_]['label'])
    kid_label = get_first_label_without_n(definitions[kid_]['label'])

    q_parent_label = f'"{parent_label}"'
    q_kid_label = f'"{kid_label}"'
    del parent_label, kid_label
    #POSTIVE
    # pairs.append((parent_label, kid_label, 'Yes'))
    # eligible_keys = [key for key in exemplars if key != hs]

    # sampled_keys = random.sample(eligible_keys, min(3, len(eligible_keys)))

    # Create a dictionary with the sampled instances
    # pairs = [exemplars[key] for key in sampled_keys]

    prompt = prefix
    # prompt = "Given two terms in a knowledge graph, your task is to determine whether they have a parent-child relationship and given a very detailed explanation on your decision."
    # for pair in pairs:

    #     random_number = random.randint(1, 10)  # Adjust the range as needed
    #     # Perform an action based on the random number
    #     if random_number % 2 == 0:
    #         prompt+= "\n\n - Question: "
    #         prompt += f"\n{pair['p'][0]} : {pair['su'][0]}"
    #         prompt += f"\n{pair['p'][1]} : {pair['su'][1]}"
    #         # Perform actions for the even case
    #         prompt+= f"\n Is {pair['p'][0]} a parent of {pair['p'][1]}?\n Answer: {'Yes' if pair['lbl'] > 0 else 'No'}" 
    #         prompt+= f"\n Explanation: \n{pair['o']}\n"

    #     else:
    #         prompt+= "\n\n - Question: "
    #         prompt += f"\n{pair['p'][1]} : {pair['su'][1]}"
    #         prompt += f"\n{pair['p'][0]} : {pair['su'][0]}"
    #         prompt+= f"\n Is {pair['p'][0]} a parent of {pair['p'][1]}?\n Answer: {'Yes' if pair['lbl'] > 0 else 'No'}"
    #         prompt+= f"\n Explanation: \n{pair['o']}\n"

    prompt+= "\n\n    - Question:"

    node_definitions = set()
    node_definitions.add(parent_)
    node_definitions.add(kid_)
    prompt += f"\n{q_parent_label} : {definitions[parent_]['summary']}"
    prompt += f"\n{q_kid_label} : {definitions[kid_]['summary']}"

    # try:
    #     for node in node_definitions:
    #         label = get_first_label_without_n(definitions[node]['label'])
    #         # logging.info(node)
    #         # logging.info(definitions[node])
    #         description = definitions[node]['summary']
    #         prompt += f"\n{label} : {description}"
    # except:
    #     print('error')
    #     continue

    prompt+= f'\n Is {q_parent_label} a parent of {q_kid_label}?\n Answer:\n' 
    # prompt+= f'\n Explanation: \n'
    # prompt+= f'\n Question: Is {get_first_label_without_n(definitions[parent_]["label"])} a parent of {get_first_label_without_n(definitions[kid_]["label"])}?\n Answer:' 
    
    prompts.append({'prompt': prompt, 
        'label': core_graph[parent_][kid_]['weight'],
        'hs': hs,
        })

    if iteration <= 10:
        logging.info(prompt)





    del hs, kid_, q_kid_label, prompt

    # NEGATIVE sample
    if random.random() < (1.5 if '0shot' in filename else 0.2):

        children = list(core_graph.neighbors(parent_))
        all_grand = set()
        for kid in children:
            # Get all grandchild nodes that are children of one child from parent_
            # For simplicity, this example assumes the graph is undirected
            grandchild_candidates = set(core_graph.neighbors(kid)) - {parent_, kid}
            all_grand = all_grand.union(grandchild_candidates)
        if not all_grand : continue
        grand_ = random.choice(list(all_grand))
        grand_label = get_first_label_without_n(definitions[grand_]['label'])
        q_grand_label = f'"{grand_label}"'
        del grand_label

        hs = HASH(definitions[parent_]['summary']+definitions[grand_]['summary'])
        



        prompt = prefix

        prompt+= "\n\n - Question: "
        node_definitions = set()
        node_definitions.add(parent_)
        node_definitions.add(grand_)
        try:
            for node in node_definitions:
                label = get_first_label_without_n(definitions[node]['label'])
                # logging.info(node)
                # logging.info(definitions[node])
                description = definitions[node]['summary']
                prompt += f"\n{label} : {description}"
        except:
            print('err')
            continue

        prompt+= f'\n Is {q_parent_label} a parent of {q_grand_label}?\n Answer:\n' 
        # prompt+= f'\n Explanation: \n'
        # prompt+= f'\n Question: Is {get_first_label_without_n(definitions[parent_]["label"])} a parent of {get_first_label_without_n(definitions[kid_]["label"])}?\n Answer:' 
        
        prompts.append({'prompt': prompt, 'label': -1, 'hs' : hs})

        # predicted_label = predict_next_token(prompt)
        if iteration <= 10:
            logging.info(prompt)
            # logging.info(predicted_label)
        edge_list_len = len(edge_list)

        if min_pair is None or edge_list_len < min_len:
            min_pair = (parent_, kid_)
            min_len = edge_list_len

        if max_pair is None or edge_list_len > max_len:
            max_pair = (parent_, kid_)
            max_len = edge_list_len
        # Check if we need to sample additional negative pairs


def save_predictions_to_file(predictions):
    with open(filename, "w") as file:
        json.dump(predictions, file, indent=4)  # Add 'indent' parameter for pretty formatting
    print(f"Predictions saved to {filename} === Total {len(predictions)}")


# PALM
def predict_batch(prompts, batch_size=10):
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
            predictions[hs] = {'i' : sentence, 'o': result}
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
                o_ids = model.generate(**i_ids, max_new_tokens=88, num_beams=10,
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
            time.sleep(0.2)


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


batch_size = 1

# predict_batch(prompts, batch_size)
# predict_llama_batch(prompts, batch_size)
# predict_gpt_batch(prompts)
# for prompt, output in zip(prompts, predictions):
#     logging.info(prompt['prompt'])
#     logging.info(output) 


