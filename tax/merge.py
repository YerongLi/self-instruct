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
import pandas as pd
from transformers import LlamaForCausalLM, AutoTokenizer, LogitsProcessorList
from torch.utils.data import DataLoader
# from datasets.dataset_dict import DatasetDict
from datasets import Dataset, DatasetDict
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
filenames=[
f"{datapath}/prediction_kshot_{TOTAL}.json",
f"{datapath}/siblings_kshot_{TOTAL}.json",
f"{datapath}/parent_kshot_{TOTAL}.json",
]

train_df = pd.DataFrame(columns=['id', 'i', 'o'])
test_df = pd.DataFrame(columns=['id', 'i', 'o'])

for filename in filenames:
    with open(filename, "r") as f:
        predictions = json.load(f)

    # Iterate through keys in predictions and append 'id', 'i', and 'o' values to DataFrames
    for idx, key in tqdm.tqdm(enumerate(predictions), total=len(predictions)):
        entry = predictions[key]
        entry['id'] = idx  # Add 'id' field with idx value
        if idx % 4 == 0:
            test_df = pd.concat([test_df, pd.DataFrame([entry])], ignore_index=True)
        else:
            train_df = pd.concat([train_df, pd.DataFrame([entry])], ignore_index=True)

# Convert DataFrames to Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Create DatasetDict
dataset_dict = DatasetDict({'train': train_dataset, 'test': test_dataset})


# train_dataset = Dataset({'id': list(range(len(dataset['train']['i']))), 'i': dataset['train']['i'], 'o': dataset['train']['o']})

# # # Create 'test' dataset
# test_dataset = Dataset({'id': list(range(len(dataset['test']['i']))), 'i': dataset['test']['i'], 'o': dataset['test']['o']})


# # # Create a DatasetDict
# dataset_dict = DatasetDict({'train': train_dataset, 'test': test_dataset})


output_file_path = f"{datapath}/dataset{TOTAL}.data"


dataset_dict.save_to_disk(output_file_path)

print(f"Dataset saved to {output_file_path}")