import argparse
import json
import random
from tqdm import tqdm

# Define your filtering values
filter_values = ["EEA", "EET", "NER", "RE"]

# Define a dictionary to keep track of the counts
value_counts = {value: 0 for value in filter_values}

def rewrite(data_batch):


    # generated_results = gptq_generate_batch(model, tokenizer, prompts)

    rewritten_batch = []
    for i, data_entry in enumerate(data_batch):
        rewritten_entry = {
            'name': data_entry['name'],
            'instruction': data_entry['instruction'] + ' ' + data_entry['schema'].replace('Text: {0}\nAnswer:', ''),
            'input': data_entry['input'],
            'output': data_entry['output']
        }
        rewritten_batch.append(rewritten_entry)
    
    return rewritten_batch

# Define and initialize data_batches here
data_batches = []

# Read the JSONL file and organize data into batches
with open('data/seed_task_ie.jsonl', 'r') as file:
    lines = file.readlines()

batch_size = 1  # Set your desired batch size
data_batches = [json.loads(line) for line in lines]
data_batches = [data_batches[i:i + batch_size] for i in range(0, len(data_batches), batch_size)]

# Create a list to store rewritten tasks
rewritten_tasks = []

with tqdm(total=len(data_batches), desc="Rewriting Tasks") as pbar:
    try:
        # Process each batch of data
        for data_batch in data_batches:
            new_rewritten_tasks = rewrite(data_batch)
            
            for task in new_rewritten_tasks:
                # Check if your filtering condition is met by using the correct key
                # Replace 'your_key' with the actual key representing the values "EEA," "EET," "NER," and "RE" in your data
                if task.get('your_key') in filter_values:
                    # Check if we have not yet sampled 100 instances for this key
                    if value_counts[task['your_key']] < 100:
                        # Perform random sampling
                        if random.random() < 0.5:  # Adjust the probability as needed
                            rewritten_tasks.append(task)
                            value_counts[task['your_key']] += 1
            
            # Update the progress bar
            pbar.update(1)

            # Check if we have sampled 100 instances for all filter values
            if all(count >= 100 for count in value_counts.values()):
                break  # Exit the loop if we have sampled 100 instances for all values

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