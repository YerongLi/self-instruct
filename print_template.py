import json
import random

# Read the JSONL file
with open('data/seed_task_ie.jsonl', 'r') as file:
    lines = file.readlines()

# Extract tasks with schema
tasks_with_schema = []
for line in lines:
    data = json.loads(line)
    if 'schema' in data:
        tasks_with_schema.append(data)

# Filter tasks with name EEA or EET
tasks_EEA_EET = [task for task in tasks_with_schema if task['name'] in ['EEA', 'EET']]

# Randomly select 10 tasks
selected_tasks = random.sample(tasks_EEA_EET, k=10)

# Generate and print examples
for task in selected_tasks:
    instruction = task['instruction']
    schema = task['schema'].replace('Text: {0}\nAnswer:', '').strip()
    input_text = task['input']
    output_text = task['output']
    
    print(f"Task: {instruction}\n\n{schema}")
    print("\nInput:", input_text)
    print("Output:")
    
    print(output_text)
    print()
