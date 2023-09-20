import argparse
import json
import os
import pandas as pd
import random
import re
import requests
import subprocess
import tqdm
import logging
from transformers import AutoTokenizer, pipeline
from auto_gptq import AutoGPTQForCausalLM
from collections import OrderedDict
# from gpt3_api import make_requests as make_gpt3_requests
from templates.instance_gen_template_ie import output_first_template_for_clf, input_first_template_for_gen
# model, tokenizer = None, None
HOST = 'localhost:5000'
URI = f'http://{HOST}/api/v1/generate'
logging.basicConfig(
    format='%(asctime)s %(levelname)-4s - %(filename)-6s:%(lineno)d - %(message)s',
    level=logging.INFO,
    filename='./output.log',
    datefmt='%m-%d %H:%M:%S')

logging.info(f'Logger start: {os.uname()[1]}')
random.seed(42)
def package(text):
        return { 'response' : {
        "id": "chatcmpl-6p9XYPYSTTRi0xEviKjjilqrWU2Ve",
        "object": "chat.completion",
        "created": 1677649420,
        "model": "gpt-3.5-turbo",
        "usage": {
            "prompt_tokens": 56,
            "completion_tokens": 31,
            "total_tokens": 87
        },
        "choices": [
            {
                "text": text,
                "finish_reason": "stop",
                "index": 0
            }
        ]
        }
    }


def run_text_gui(prompt):
    logging.info('Prompt :')
    logging.info(prompt)
    request = {
        'prompt': prompt,
        'max_new_tokens': 1000,
        'auto_max_new_tokens': False,
        'max_tokens_second': 0,

        # Generation params. If 'preset' is set to different than 'None', the values
        # in presets/preset-name.yaml are used instead of the individual numbers.
        'preset': 'None',
        'do_sample': True,
        'temperature': 0.7,
        'top_p': 0.1,
        'typical_p': 1,
        'epsilon_cutoff': 0,  # In units of 1e-4
        'eta_cutoff': 0,  # In units of 1e-4
        'tfs': 1,
        'top_a': 0,
        'repetition_penalty': 1.18,
        'repetition_penalty_range': 0,
        'top_k': 40,
        'min_length': 0,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'mirostat_mode': 0,
        'mirostat_tau': 5,
        'mirostat_eta': 0.1,
        'guidance_scale': 1,
        'negative_prompt': '',

        'seed': -1,
        'add_bos_token': True,
        'truncation_length': 4096,
        'ban_eos_token': False,
        'skip_special_tokens': True,
        'stopping_strings': []
    }

    try:
        response = requests.post(URI, json=request)
        if response.status_code == 200:
            result = response.json()['results'][0]['text']
            logging.info('Result :')
            logging.info(result)
            return result
    except Exception as e:
        print(f"An error occurred: {e}")

def run_llama_command(input_string):
    def process_output(output):
        # Define the command as a list of individual components
        command = [
            "$SCRATCH/llama.cpp/main",
            "-m",
            "$SCRATCH/.cache/pyllama/Llama-2-70b/ggml-model-q4_0.gguf",
            "-p",
            f'"{input_string}"',  # Wrap input_string with double quotes
            "-t",
            "1",
            "-n",
            "512",
            "--temp",
            "0.1",
            "--top-p",
            "0.90",
            "-ngl",
            "100"
        ]

        # Join the command list into a single string with spaces
        command_str = " ".join(command)

        try:
            result = subprocess.run(command_str, shell=True, check=True, capture_output=True, text=True)

            return process_output(result.stdout)[1]

        except subprocess.CalledProcessError as e:
            return f"Error executing the command: {e}"

def query(prompt, option='text'):
    if option == 'text':
        return run_text_gui(prompt)
    elif option == 'cpp':
        return run_llama_command(prompt)
    else:
        raise ValueError("Invalid option. Supported options are 'text' and 'cpp'")

def extract_prefix_until_index(input_string):
    lines = input_string.split('\n')  # Split the string into lines

    output = []

    for line in lines:
        # Use regular expression to check if the line starts with an index
        if re.match(r'^\d+\.\s', line):
            break
        output.append(line)

    result = '\n'.join(output)
    return result.strip()
def remove_prefix_markers(input_string, end_marker):
    end_index = input_string.find(end_marker)
    if end_index != -1:
        extracted_text = input_string[end_index + len(end_marker):].strip()
        return extracted_text
    else:
        return "Markers not found in the input string."

def run_llama_command(input_string, gpt3=True):
    input_string = re.sub(r'(?<!\\)"', r'\\"', input_string)
    if not gpt3:
        # Define the command as a list of individual components


        # command = [
        #     "$SCRATCH/llama.cpp/main",
        #     "-m",
        #     "$SCRATCH/.cache/pyllama/Llama-2-7b/ggml-model-q4_0.gguf",
        #     "-p",
        #     f'"{input_string}"',  # Wrap input_string with double quotes
        #     "-t",
        #     "1",
        #     "-n",
        #     "4096",
        #     "--temp",
        #     "0.1",
        #     "--top-p",
        #     "0.90",
        #     "-ngl",
        #     "83"
        # ]

        command = [
            "$SCRATCH/llama.cpp/main",
            "-m",
            "$SCRATCH/.cache/pyllama/Llama-2-70b/ggml-model-q4_0.gguf",
            "-p",
            f'"{input_string}"',  # Wrap input_string with double quotes
            "-t",
            "1",
            "-n",
            "4096",  # Reduced from 2048
            "--temp",
            "0.1",
            "--top-p",
            "0.90",
            "-ngl",
            "100",  # Changed from 83
        ]

        # Join the command list into a single string with spaces
        command_str = " ".join(command)

        try:
            result = subprocess.run(command_str, shell=True, check=True, capture_output=True, text=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            print('Error')
            return f"Error executing the command: {e}"
    else:
        # Return GPT-3 format response
        return package(run_llama_command(input_string, False))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        required=True,
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="machine_generated_instructions.jsonl"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="machine_generated_instances.jsonl",
    )
    parser.add_argument(
        "--num_instructions",
        type=int,
        help="if specified, only generate instance input for this many instructions",
    )
    parser.add_argument(
        "--max_instances_to_generate",
        type=int,
        default=5,
        help="The max number of instances to generate for each instruction.",
    )
    parser.add_argument(
        "--generation_tasks_only",
        action="store_true",
        help="If specified, only do for generation tasks.",
    )
    parser.add_argument(
        "--classification_tasks_only",
        action="store_true",
        help="If specified, only do for classification tasks.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="davinci",
        help="The engine to use."
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=5,
        help="The number of requests to send in a batch."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="The API key to use. If not specified, the key will be read from the environment variable OPENAI_API_KEY."
    )
    parser.add_argument(
        "--organization",
        type=str,
        help="The organization to use. If not specified, the default organization id will be used."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="A boolean flag for testing."
    )
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()

    with open(os.path.join(args.batch_dir, args.input_file)) as fin:
        lines = fin.readlines()
        if args.num_instructions is not None:
            lines = lines[:args.num_instructions]
        tasks = []
        for line in lines:
            data = json.loads(line)
            if "metadata" in data:
                data["instruction_metadata"] = data["metadata"]
                del data["metadata"]
            tasks.append(data)

    # task_clf_types = {}
    # with open(os.path.join(args.batch_dir, "is_clf_or_not_davinci_template_1.jsonl")) as fin:
    #     for line in fin:
    #         data = json.loads(line)
    #         task_clf_types[data["instruction"]] = data["is_classification"].strip() in ["Yes", "yes", "YES"]

    # if args.classification_tasks_only:
    #     tasks = [task for task in tasks if task_clf_types[task["instruction"]]]
    
    # if args.generation_tasks_only:
    #     tasks = [task for task in tasks if not task_clf_types[task["instruction"]]]

    output_path = os.path.join(args.batch_dir, args.output_file)
    existing_requests = {}
    if os.path.exists(output_path):
        with open(output_path) as fin:
            for line in tqdm.tqdm(fin):
                try:
                    data = json.loads(line)
                    existing_requests[data["instruction"]] = data
                except:
                    pass
        print(f"Loaded {len(existing_requests)} existing requests")

    progress_bar = tqdm.tqdm(total=len(tasks))
    with open(output_path, "w") as fout:
        for batch_idx in range(0, len(tasks), args.request_batch_size):
            batch = tasks[batch_idx: batch_idx + args.request_batch_size]
            if all(d["instruction"] in existing_requests for d in batch):
                for d in batch:
                    data = existing_requests[d["instruction"]]
                    data = OrderedDict(
                        (k, data[k]) for k in \
                            ["instruction", "format", "instance_metadata", "instruction_metadata", 
                            "most_similar", "avg_similarity_score",'instance']
                        )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            else:
                prompts = []
                for task in batch:
                    # if task_clf_types[task["instruction"]]:
                    #     prompt = output_first_template_for_clf + " " + task["instruction"].strip() + "\n"
                    #     prompts.append(prompt)
                    # else:
                        prompt = input_first_template_for_gen + " " + task["instruction"].strip() + "\nFormat:"
                        prompts.append(prompt)

                end_marker = input_first_template_for_gen[-160:] ## TODO remove the prefix

                results = [
                    package(query(prompt)) for prompt in prompts
                ]

                # Example usage
                # input_string = "Yann LeCun, Yoshua Bengio\nOutput: Alan Turing\n\nTask: Extract information from text."


                # print(type(input_first_template_for_gen))
                # logging.info('end_marker')

                # print(end_marker)


                # for prompt, result in zip(prompts, results):
                #     logging.info(f"Prompt: {prompt}")
                #     logging.info(f"Result: {result}")
                #     logging.info("-" * 30)  # Separator between pairs
                # results = make_gpt3_requests(
                #     engine=args.engine,
                #     prompts=prompts,
                #     # because the clf template is longer, we need to decrease the max_tokens
                #     max_tokens=300 if any(task_clf_types[task["instruction"]] for task in batch) else 350,
                #     temperature=0,
                #     top_p=0,
                #     frequency_penalty=0,
                #     presence_penalty=1.5,
                #     stop_sequences=[f"Example {args.max_instances_to_generate + 1}", "Task:"],
                #     logprobs=1,
                #     n=1,
                #     best_of=1,
                #     api_key=args.api_key,
                #     organization=args.organization)
                for i in range(len(batch)):
                    data = batch[i]
                    data["instance_metadata"] = results[i]
                    if results[i]["response"] is not None:
                        data["instance"] = extract_prefix_until_index(
                            results[i]["response"]["choices"][0]["text"]
                            )
                    else:
                        data["format"] = ""
                    data = OrderedDict(
                        (k, data[k]) for k in \
                            ["instruction", "format", "instance_metadata", "instruction_metadata", 
                            "most_similar", "avg_similarity_score",'instance']
                        )
                    del data['instance_metadata'] # TODO remove
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            progress_bar.update(len(batch))
