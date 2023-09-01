import json

# Define the rewrite function
def remove_prefix_markers(input_string, end_marker):
    end_index = input_string.find(end_marker)
    if end_index != -1:
        extracted_text = input_string[end_index + len(end_marker):].strip()
        return extracted_text
    else:
        return "Markers not found in the input string."

def gptq_generate(model, tokenizer, input_text, max_tokens=4096, temperature=0.7, top_p=0.95, repetition_penalty=1.15):
    prompt_template = f"{input_text}\n"
    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.to(model.device)
    output = model.generate(input_ids=input_ids, temperature=temperature, max_length=max_tokens, top_p=top_p, repetition_penalty=repetition_penalty)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def rewrite(schema, input_text, output_text):
    full_text = f"{schema}\n{input_text}\n{output_text}"
    return remove_prefix_markers(gptq_generate(model, tokenizer, full_text), "Markers not found in the input string.")
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
model = AutoGPTQForCausalLM.from_quantized(model_name_or_path, model_basename="model", inject_fused_attention=False, use_safetensors=True, trust_remote_code=False, device_map="auto", use_triton=False, quantize_config=None)

# Read the JSONL file
with open('data/seed_task_ie.jsonl', 'r') as file:
    lines = file.readlines()

# Create a list to store rewritten tasks
rewritten_tasks = []

# Process each line in the JSONL file
for line in lines:
    data = json.loads(line)
    
    if 'schema' in data:
        instruction = data['instruction']
        schema = data['schema'].replace('Text: {0}\nAnswer:', '')
        input_text = data['input']
        output_text = data['output']
        
        # Use the rewrite function to add the "rewrite" prefix to schema
        rewritten_schema = rewrite(schema)
        
        # Create a dictionary for the rewritten task
        rewritten_task = {
            'instruction': instruction,
            'schema': rewritten_schema,
            'input': input_text,
            'output': output_text
        }
        
        # Append the rewritten task to the list
        rewritten_tasks.append(rewritten_task)

# Write all rewritten tasks to a new file
with open('data/rewrite_seed_task_ie.jsonl', 'w') as rewrite_file:
    for task in rewritten_tasks:
        rewrite_file.write(json.dumps(task) + '\n')

print('All tasks have been rewritten and stored in "data/rewrite_seed_task_ie.jsonl"')
