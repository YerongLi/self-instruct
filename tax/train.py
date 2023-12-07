import logging
import os
from datasets import load_dataset
from datasets import concatenate_datasets
import numpy as np
import tqdm
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import DataCollatorForSeq2Seq, TrainerCallback
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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
from datasets import DatasetDict
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
parser.add_argument("--r", default=None, type=str,
                        help="Path to the checkpoint to resume training. Default is 'results/best'.")

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
LOGFILE='output.log'
logging.basicConfig(
    format='%(asctime)s %(levelname)-4s - %(filename)-6s:%(lineno)d - %(message)s',
    level=logging.INFO,
    filename=LOGFILE,
    datefmt='%m-%d %H:%M:%S')
# input_header='dialogue'
# output_header='summary'
input_header='i'
output_header='o'
logging.info(f'Logger start: {os.uname()[1]}')
# Load dataset from the hub

# dataset = load_dataset("samsum")
dataset= DatasetDict.load_from_disk(f"{datapath}/dataset{TOTAL}.data")


model_id='/scratch/yerong/.cache/pyllama/flan-t5-base'
logging.info(model_id)
# Define LoRA Config
lora_config = LoraConfig(
 r=16,
 lora_alpha=32,
 target_modules=["q", "v"],
 lora_dropout=0.05,
 bias="none",
 task_type=TaskType.SEQ_2_SEQ_LM
)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto")

# add LoRA adaptor
checkpoint_to_resume= args.r
if checkpoint_to_resume:
        print('Loading checkpoint')
        model = PeftModel.from_pretrained(model, checkpoint_to_resume, is_trainable=True)
else:

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
# print(dataset)
# print(type(dataset))
class SaveBestModelCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.best_eval_loss = float("inf")

    def on_evaluate(self, args, state, control, model, tokenizer, **kwargs):
        eval_loss = kwargs['metrics']['eval_loss']
        # print(self.output_dir)
        # print(eval_loss, self.best_eval_loss)
        if eval_loss < self.best_eval_loss:
            # Save the model if the evaluation loss improves
            model.save_pretrained(f"{self.output_dir}/best/")
            self.best_eval_loss = eval_loss
            logging.info(f"Model saved with eval loss: {eval_loss}")


print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")
# logging.info(dataset['train'])
for i in tqdm.tqdm(range(len(dataset['train'][input_header][:20]))):
	# dataset['train'][input_header][i] = 10000 *'Here it is ' +  dataset['train'][input_header][i]
	logging.info(dataset['train'][input_header][i])
	logging.info(dataset['train'][output_header][i])
# Train dataset size: 14732
# Test dataset size: 819

# model_id="google/flan-t5-xl"

tokenizer = AutoTokenizer.from_pretrained(model_id)
# The maximum total input sequence length after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded.

# tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x[input_header], truncation=True), batched=True, remove_columns=[input_header, output_header])
tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x[input_header], truncation=False), batched=True, remove_columns=[input_header, output_header])
input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
# take 85 percentile of max length for better utilization
max_source_length = int(np.percentile(input_lenghts, 85))
print(f"Max source length: {max_source_length}")

# The maximum total sequence length for target text after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded."
# tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x[output_header], truncation=True), batched=True, remove_columns=[input_header, output_header])
tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x[output_header], truncation=False), batched=True, remove_columns=[input_header, output_header])
target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
# take 90 percentile of max length for better utilization
max_target_length = int(np.percentile(target_lenghts, 90))
print(f"Max target length: {max_target_length}")


def preprocess_function(sample,padding="max_length"):
    # add prefix to the input for t5
    inputs = ["summarize: " + item for item in sample[input_header]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample[output_header], max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=[input_header, output_header, "id"])
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

# save datasets to disk for later easy loading
tokenized_dataset["train"].save_to_disk("data/train")
tokenized_dataset["test"].save_to_disk("data/eval")






# trainable params: 18874368 || all params: 11154206720 || trainable%: 0.16921300163961817



# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
output_dir = './results'
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
	auto_find_batch_size=True,
    learning_rate=1e-3, # higher learning rate
    num_train_epochs=5,
    save_steps=100,
    save_total_limit=2,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=30,
    do_eval=True,
    evaluation_strategy='steps',
    eval_steps=200,
    # save_strategy="no",
)

save_best_model_callback = SaveBestModelCallback(output_dir=training_args.output_dir)

tokenized_dataset['train'] = tokenized_dataset['train'].shuffle(seed=42)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    callbacks=[save_best_model_callback],  # Include the callback here
)
model.config.use_cache = False
# train model
trainer.train()

