import argparse
import json
import logging
import os
import pickle
import random
import shutil
import tqdm

parser = argparse.ArgumentParser(description="Your script description")

# Add the configuration file argument
parser.add_argument("config_file", type=str, help="Path to the configuration file")

args = parser.parse_args()
config_file = args.config_file

# Read the configuration file
with open(config_file) as f:
    config = json.load(f)

# Get the configuration values
TOTAL = config['TOTAL']
taxofilename = config['taxofilename']
source_file = config['source_term_file']
destination_file = config['destination_file']

# Read the paths to the pickle files from the configuration file
all_path_pkl = config['all_path_pkl']
definitions_pkl = config['definitions_pkl']
edges_pkl = config['edges_pkl']

# Copy the source file to the destination file
shutil.copyfile(source_file, destination_file)

# Set up logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-4s - %(filename)-6s:%(lineno)d - %(message)s',
    level=logging.INFO,
    filename='./output.log',
    datefmt='%m-%d %H:%M:%S')

logging.info(f'Logger start: {os.uname()[1]}')

# Load the data from the pickle files
with open(all_path_pkl, 'rb') as f:
    all_path = pickle.load(f)

with open(definitions_pkl, 'rb') as f:
    definitions = pickle.load(f)

with open(edges_pkl, 'rb') as f:
    edges = pickle.load(f)
edges_dict = {pair: 0 for pair in list(edges)}
logging.info(len(edges_dict))
all_paths = [[] for _ in range(20)]
for path in all_path:
    all_paths[len(path)].append(path)
# logging.info(len(all_paths[3]))
# logging.info(len(all_paths[4]))
for i in range(len(all_path)):
    # logging.info('====== path ===========')
    for term in all_path[i]:
        # logging.info(definitions[term])
        pass

ans = {}
def sample_strategy_1(ans, all_paths):
    # skip a level in paths of length 3
    for _ in range(1000):
        path_length = 3
        path = random.choice(all_paths[path_length])
        first = path[0]
        third = path[2]
        pair = (first, third)
        if (not pair in ans) and (not pair in edges_dict):
            ans[(first, third)] = 1
            return True
    return False

def sample_strategy_2(ans, all_paths):
    # skip a level in paths of length 4
    for _ in range(1000):
        path_length = 4
        path = random.choice(all_paths[path_length])
        second = path[1]
        fourth = path[3]
        pair = (second, fourth)
        if (not pair in ans) and (not pair in edges_dict):
            ans[(second, fourth)] = 2
            return True
    return False

def sample_strategy_3(ans, all_paths):
    # re-root to a wrong parent
    for _ in range(1000):
        path_length = 3
        path = random.choice(all_paths[path_length])
        third = path[2]

        # Find a path that does not contain the third item
        valid_path = False
        while not valid_path:
            pp = random.choice(all_paths[path_length])
            if third not in pp:
                first = pp[0]
                valid_path = True
        pair = (first, third)
        if (not pair in ans) and (not pair in edges_dict):
            ans[(first, third)] = 3
            return True
    return False

def expand_ans(ans, all_paths):
    # Assign weights to each strategy
    strategy_weights = {
        sample_strategy_1: 0.333333,
        sample_strategy_2: 0.333333,
        sample_strategy_3: 1- 0.333333*2
    }

    # Randomly select a strategy based on weights
    strategy = random.choices(list(strategy_weights.keys()), weights=list(strategy_weights.values()))[0]

    # Execute the selected strategy
    if strategy(ans, all_paths):
        return True
    return False

# Expand the ans dictionary until it reaches a size of TOTAL
progress_bar = tqdm.tqdm(range(TOTAL))
while True:
    if expand_ans(ans, all_paths):
        progress_bar.update()
        if len(ans) == TOTAL:
            progress_bar.close()
            break
    else:
        progress_bar.refresh()


print("Size of ans dictionary:", len(ans))

for pair in ans:
    logging.info(f"========= {ans[pair]} ==========")
    logging.info(definitions[pair[0]])
    logging.info(definitions[pair[1]])

with open('../../TaxoComplete/data/SemEval-Noun/wordnet_noun.taxo', 'r') as fin, open(taxofilename, 'w') as fout:
    for line in fin:
        fout.write(line.strip() + '\t1\n')

def get_first_label_with_n(label_str):
    # Split the label string by "||"
    labels = label_str.split('||')

    # Find the first label that contains ".n."
    for label in labels:
        if ".n." in label:
            return label

    # If no label contains ".n.", return the first label
    return labels[0]

# Extract and write the first labels with ".n." to the output file
with open(taxofilename, 'a') as fout:
    for pair in ans:
        first_label_0 = get_first_label_with_n(definitions[pair[0]]['label'])
        first_label_1 = get_first_label_with_n(definitions[pair[1]]['label'])
        fout.write(f"{first_label_0}\t{first_label_1}\t-1\n")
