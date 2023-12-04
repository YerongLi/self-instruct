import json
import random

# Read the JSONL file
with open('data/seed_task_ie.jsonl', 'r') as file:
    lines = file.readlines()

# Extract tasks with schema and names 'EEA' or 'EET'
tasks_with_schema_EEA_EET = [data for line in lines if (data := json.loads(line)) and 'schema' in data and data['name'] in ['EEA', 'EET']]

# Sample 4 tasks from 'EEA' and 'EET'
selected_tasks_EEA_EET = random.sample(tasks_with_schema_EEA_EET, k=4)

# Extract tasks with schema
tasks_with_schema = [data for line in lines if (data := json.loads(line)) and 'schema' in data]

# Sample 6 tasks randomly from the remaining tasks
selected_tasks_global = random.sample(tasks_with_schema, k=6)

# Generate and print examples for selected tasks
selected_tasks = selected_tasks_EEA_EET + selected_tasks_global
for idx, task in enumerate(selected_tasks):
    instruction = task['instruction']
    schema = task['schema'].replace('Text: {0}\nAnswer:', '')
    input_text = task['input']
    output_text = task['output']
    
    print(f"Task: {instruction} {schema}")
    print("Input:", input_text)
    print("Output:", output_text)
    print()
print('Task:')
