import argparse
import json
import random
from tqdm import tqdm  # Import the tqdm library

# Define your filtering values
filter_values = ["EEA", "EET", "NER", "RE"]

# Define a dictionary to keep track of the counts
value_counts = {value: 0 for value in filter_values}

# ... (The rest of your code remains the same up to the data processing part)

# Create a list to store rewritten tasks
rewritten_tasks = []

# Create a progress bar for processing batches of data
with tqdm(total=len(data_batches), desc="Rewriting Tasks") as pbar:
    try:
        # Process each batch of data
        for data_batch in data_batches:
            new_rewritten_tasks = rewrite(data_batch)
            
            for task in new_rewritten_tasks:
                # Check if the 'name' value is in the filter_values
                if task['name'] in filter_values:
                    # Check if we have not yet sampled 100 instances for this 'name' value
                    if value_counts[task['name']] < 100:
                        rewritten_tasks.append(task)
                        value_counts[task['name']] += 1
            
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
