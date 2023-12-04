
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
# import openai
import re
import google.generativeai as palm
import os

from transformers import LlamaForCausalLM, AutoTokenizer, LogitsProcessorList
from torch.utils.data import DataLoader, Dataset
from openai import OpenAI
LOGFILE='output.log'
logging.basicConfig(
    format='%(asctime)s %(levelname)-4s - %(filename)-6s:%(lineno)d - %(message)s',
    level=logging.INFO,
    filename=LOGFILE,
    datefmt='%m-%d %H:%M:%S')

logging.info(f'Logger start: {os.uname()[1]}')
palm.configure(api_key=os.environ['PALM'])
# from transformers import LlamaForCausalLM, AutoTokenizer, LogitsProcessorList
# from torch.utils.data import DataLoader, Dataset
parser = argparse.ArgumentParser(description="Your script description")
# Add the configuration file argument
parser.add_argument("config_file", type=str, help="Path to the configuration file")
# parser.add_argument("TOTAL", type=int, default=700, nargs="?", help="Number of total items to process")
# parser.add_argument("mode", type=str, default='llama', nargs="?", help="Prediction mode")

def HASH(input_string):
    # Use SHA-256 for deterministic hashing
    hash_object = hashlib.sha256(input_string.encode())
    hash_value = int.from_bytes(hash_object.digest(), byteorder='big')

    return str(hash_value)
args = parser.parse_args()

config_file = args.config_file
# Read the configuration file
with open(config_file) as f:
    config = json.load(f)

# Get the configuration values
# Extract the base filename without the ".taxo" extension
datapath = config['taxofilename'].split('/')[:-1]
datapath = '/'.join(datapath)
print(datapath)


# Define the function to process each JSON file
def process_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    # Iterate over items in the dictionary
    for item_key, item_value in data.items():
        if isinstance(item_value, dict):
            # Extract the relevant portions from "i"
            i_text = item_value.get("i", "")
            match = re.search(r"Answer:\s*Yes|Answer:\s*No", i_text)

            if match:
                cut_off_index = match.start()
                i_cut = i_text[:cut_off_index].strip()
                o_text = f"Answer:\n{match.group().strip()}\n\nExplanation:\n{i_text[cut_off_index + len(match.group()):].strip()}"
                
                # Update the "i" and "o" fields in the item
                item_value["i"] = i_cut
                item_value["o"] = o_text
                data[item_key] = item_value
                logging.info(data[item_key]["i"])

    # Construct the new file name by appending 'r' to the original file name
    new_file_path = f"{file_path}r"
    # new_file_path = os.path.join(datapath, new_file_name)
    print(new_file_path)

    # # Save the modified data to the new file
    # with open(new_file_path, 'w', encoding='utf-8') as new_json_file:
    #     json.dump(data, new_json_file, ensure_ascii=False, indent=2)

# Set the directory path containing the JSON files

# Iterate through all JSON files in the directory
for filename in os.listdir(datapath):
    if filename.endswith(".json"):
        filename='../../TaxoComplete/data/SemEval-Noun/siblings_0shot_300.json'
        file_path = os.path.join(datapath, filename)
        process_json_file(file_path)
        break
