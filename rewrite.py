import argparse
import json
import torch
from auto_gptq import AutoGPTQForCausalLM
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline
# Define the rewrite function
def remove_prefix_markers(input_string, end_marker):
    end_index = input_string.find(end_marker)
    if end_index != -1:
        extracted_text = input_string[end_index + len(end_marker):].strip()
        return extracted_text
    else:
        return "Markers not found in the input string."

def gptq_generate_batch(model, tokenizer, input_texts, max_tokens=4096, temperature=0.7, top_p=0.95, repetition_penalty=1.15):
    # Encode the input_texts in batch
    encoding = tokenizer(input_texts, padding=True, return_tensors='pt').to(model.device)
    
    # Generate text in batch using the model
    with torch.no_grad():
        generated_ids = model.generate(**encoding, max_length=max_tokens, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty)
    
    # Decode the generated IDs into text
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    return generated_texts

def rewrite(schema, input_text, output_text):
    prompt = f"""
There are four other different ways to write the instruction in the following information extraction question (with a possibly different required output format):
INST {schema}
INPUT {input_text}
OUTPUT {output_text}

(1) 
"""
    print(prompt)
    return [remove_prefix_markers(rs, prompt[:20]) for rs in gptq_generate_batch(model, tokenizer, [prompt])]

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="A boolean flag for testing."
    )
    return parser.parse_args()
args = parse_args()
# model_name_or_path = "/scratch/yerong/.cache/pyllama/Llama-2-70B-GPTQ"
if args.test:
    model_name_or_path = "/scratch/yerong/.cache/pyllama/Llama-2-7B-GPTQ"
else:
    model_name_or_path = "/scratch/yerong/.cache/pyllama/Llama-2-70B-GPTQ"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoGPTQForCausalLM.from_quantized(model_name_or_path, model_basename="model", inject_fused_attention=False, use_safetensors=True, trust_remote_code=False, device_map="auto", use_triton=False, quantize_config=None)

# Read the JSONL file
with open('data/seed_task_ie.jsonl', 'r') as file:
    lines = file.readlines()

# Create a list to store rewritten tasks
rewritten_tasks = []

# Create a progress bar for processing lines in the JSONL file
with tqdm(total=len(lines), desc="Rewriting Tasks") as pbar:
    try:
        # Process each line in the JSONL file
        for line in lines:
            data = json.loads(line)
            
            if 'schema' in data:
                instruction = data['instruction']
                schema = data['schema'].replace('Text: {0}\nAnswer:', '')
                input_text = data['input']
                output_text = data['output']
                
                # Use the rewrite function to add the "rewrite" prefix to schema
                rewritten_schema = rewrite(schema, input_text, output_text)
                
                # Create a dictionary for the rewritten task
                rewritten_task = {
                    'instruction': instruction,
                    'schema': rewritten_schema,
                    'input': input_text,
                    'output': output_text
                }
                
                # Append the rewritten task to the list
                rewritten_tasks.append(rewritten_task)
                
                # Update the progress bar
                pbar.update(1)
    
    except KeyboardInterrupt:
        # Handle keyboard interrupt (Ctrl+C)
        print("\nKeyboard interrupt detected. Saving progress to 'data/rewrite_seed_task_ie.jsonl'...")
        
        # Write the tasks processed so far to the rewrite file
        with open('data/rewrite_seed_task_ie.jsonl', 'w') as rewrite_file:
            for task in rewritten_tasks:
                rewrite_file.write(json.dumps(task) + '\n')
        
        print('Progress saved. Exiting.')

# Write all rewritten tasks to a new file
with open('data/rewrite_seed_task_ie.jsonl', 'w') as rewrite_file:
    for task in rewritten_tasks:
        rewrite_file.write(json.dumps(task) + '\n')

print('All tasks have been rewritten and stored in "data/rewrite_seed_task_ie.jsonl"')