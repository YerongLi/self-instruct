import argparse
import json
import signal
from tqdm import tqdm
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

# ... (other code remains the same as in your provided code)

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
