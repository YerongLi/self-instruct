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

# Randomly select 10 tasks
selected_tasks = random.sample(tasks_with_schema, k=10)

# Generate and print examples
for idx, task in enumerate(selected_tasks):
    task_name = task['name']
    schema = task['schema']
    output_format_start = "Output format should be \""
    output_format_end = "\"."
    
    output_format_start_idx = schema.find(output_format_start)
    output_format_end_idx = schema.find(output_format_end, output_format_start_idx)
    
    if output_format_start_idx != -1 and output_format_end_idx != -1:
        output_format = schema[output_format_start_idx + len(output_format_start):output_format_end_idx]
        output_parts = [part.strip() for part in output_format.split(';')[0].split(':')]
        
        print(f"Task {idx + 1}: {task_name}")
        print("Output:")
        for word in output_parts[1:]:
            print(f"- {word}")
        print()
