import logging
import os
import pickle
import random
import tqdm
# import networkx as nx
import torch
from transformers import LlamaForCausalLM, AutoTokenizer, LogitsProcessorList
from torch.utils.data import DataLoader, Dataset

TOTAL = 700
logging.basicConfig(
    format='%(asctime)s %(levelname)-4s - %(filename)-6s:%(lineno)d - %(message)s',
    level=logging.INFO,
    filename='./output.log',
    datefmt='%m-%d %H:%M:%S')

logging.info(f'Logger start: {os.uname()[1]}')

model_path = "/scratch/yerong/.cache/pyllama/Llama-2-7b-hf/"

# model = None
model = LlamaForCausalLM.from_pretrained(
  model_path,
  torch_dtype=torch.float16,
  device_map='auto',
).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = "[PAD]"
tokenizer.padding_side = "left"
device = "cuda:0" # You can set this to "cpu" if you don't have a GPU
logits_processor = LogitsProcessorList()
# logging.info(f'Yes id is : {tokenizer(["Yes"])}')
# logging.info(f'No id is : {tokenizer(["No"])}')
# 11-27 02:16:11 INFO - query1.py:28 - Yes id is : {'input_ids': [[1, 3869]], 'attention_mask': [[1, 1]]}
# 11-27 02:16:11 INFO - query1.py:29 - No id is : {'input_ids': [[1, 1939]], 'attention_mask': [[1, 1]]}
def collate_fn(batch):
    # Get the input_ids and attention_masks from the batch
    input_ids = [item[0] for item in batch]
    attn_masks = [item[1] for item in batch]

    # Calculate the maximum length in the batch
    max_len = max([input_id.shape[1] for input_id in input_ids])

    # Pad input_ids and attention_masks
    padded_input_ids = []
    padded_attn_masks = []

    for input_id, attn_mask in zip(input_ids, attn_masks):
        if input_id.shape[1] < max_len:
            padding = torch.zeros((1, max_len - input_id.shape[1])).to(input_id.dtype)
            padded_input_id = torch.cat([input_id, padding], dim=1)
        else:
            padded_input_id = input_id

        if attn_mask.shape[1] < max_len:
            padding = torch.zeros((1, max_len - attn_mask.shape[1])).to(attn_mask.dtype)
            padded_attn_mask = torch.cat([attn_mask, padding], dim=1)
        else:
            padded_attn_mask = attn_mask

        padded_input_ids.append(padded_input_id)
        padded_attn_masks.append(padded_attn_mask)

    return torch.stack(padded_input_ids, dim=0), torch.stack(padded_attn_masks, dim=0)


class PromptDataset(Dataset):
    def __init__(self, prompts, tokenizer, max_length=None):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        encoding = self.tokenizer(prompt, return_tensors="pt", padding=True, max_length=self.max_length)
        logging.info(encoding)
        logging.info(len(encoding['input_ids']))
        logging.info(len(encoding['attention_mask']))
        return encoding

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

def predict_next_token_batch(prompts, batch_size=10):
    predictions = []
    sentences = [item['prompt'] for item in prompts]
    # Split prompts into batches
    for i in tqdm.tqdm(range(0, len(sentences), batch_size), desc="Processing Batches", unit="batch"):
        batch_prompts = sentences[i:i + batch_size]

        # Tokenize prompts and convert to PyTorch tensors
        input_ids = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)

        # Generate logits for the next token using the model
        with torch.no_grad():
            outputs = model(**input_ids)
            logits = outputs.logits[:, -1, :]

        # Process logits or do whatever you need with them
        next_tokens_scores = logits  # Assuming logits_processor is not used in this function
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)
        # Example: Extract probabilities for specific tokens (adjust token IDs as needed)
        yes_prob = next_tokens_scores[:, 3869]
        no_prob = next_tokens_scores[:, 1939]

        # Calculate the difference in probabilities
        prob_diff = yes_prob - no_prob

        # Determine the predictions based on probability differences
        batch_predictions = torch.where(prob_diff > 0, 1, -1).tolist()
    
        # Append batch predictions to the overall predictions list
        predictions.extend(batch_predictions)

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
with open(f'../../TaxoComplete/core_graph_{TOTAL}.pkl', 'rb') as f:
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
count_edges = 0
count_neg_label = 0
prompts = []

# for iteration, edge in tqdm.tqdm(enumerate(list(core_graph.edges())[:31]), total=31):
for iteration, edge in tqdm.tqdm(enumerate(core_graph.edges()), total=core_graph.number_of_edges()):
    parent_, kid_ = edge
    if parent_ == rootkey or kid_ == rootkey : continue
    count_edges+= 1
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
    
    prompts.append({'prompt': prompt, 'label': core_graph[parent_][kid_]['weight']})
    # predicted_label = predict_next_token(prompt)
    if iteration <= 0:
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

batch_size = 32


# # Create a dataset and dataloader
# prompt_dataset = PromptDataset(prompts, tokenizer, max_length=max_length)
# prompt_dataloader = DataLoader(prompt_dataset, batch_size=batch_size, shuffle=False, num_workers=1)  # Adjust num_workers based on your system capabilities

# # Iterate over batches
# for batch in prompt_dataloader:
#     inputs = {k: v.to(device) for k, v in batch.items()}
    
#     with torch.no_grad():
#         outputs = model(**inputs)

#     logits = outputs.logits
#     # Process logits or do whatever you need with them
#     print(logits.shape)


predictions =predict_next_token_batch(prompts, batch_size)
print(len(predictions))
print(len(prompts))
result = [{'label':prompts[i]['label'], 'pred': predictions[i]} for i in range(len(prompts))]
# output_sequences = model.generate(**inputs, max_new_tokens=20, do_sample=True, top_p=0.9)

# print(tokenizer.batch_decode(output_sequences, skip_special_tokens=True))
logging.info(f"Count number of -1 {count_neg_label}")

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












from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, confusion_matrix

# Extract ground truth and predicted labels
ground_truth = [label_dict['label'] for label_dict in result]
predicted_labels = [label_dict['pred'] for label_dict in result]

# Calculate F1 score
f1_score_value = f1_score(ground_truth, predicted_labels, average='macro')

if isinstance(f1_score_value, tuple):
    # Handle tuple output
    f1 = f1_score_value[0]
    logging.info("F1 score: %f", f1)
else:
    # Handle single value output
    f1 = f1_score_value
    logging.info("F1 score: %f", f1)

# Calculate accuracy score
accuracy_score_value = accuracy_score(ground_truth, predicted_labels)

if isinstance(accuracy_score_value, tuple):
    # Handle tuple output
    accuracy = accuracy_score_value[0]
    logging.info("Accuracy score: %f", accuracy)
else:
    # Handle single value output
    accuracy = accuracy_score_value
    logging.info("Accuracy score: %f", accuracy)

# Calculate recall score
recall_score_value = recall_score(ground_truth, predicted_labels, average='macro')

if isinstance(recall_score_value, tuple):
    # Handle tuple output
    recall = recall_score_value[0]
    logging.info("Recall score: %f", recall)
else:
    # Handle single value output
    recall = recall_score_value
    logging.info("Recall score: %f", recall)

# Calculate AUC score
auc_score_value = roc_auc_score(ground_truth, predicted_labels)

if isinstance(auc_score_value, tuple):
    # Handle tuple output
    auc = auc_score_value[0]
    logging.info("AUC score: %f", auc)
else:
    # Handle single value output
    auc = auc_score_value
    logging.info("AUC score: %f", auc)

# Print confusion matrix
conf_matrix = confusion_matrix(ground_truth, predicted_labels, labels=[1, -1])
logging.info("Confusion matrix:")
print(conf_matrix)