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
    output_parts = [part.strip() for part in schema.split('Output format should be "')[1].split(';')[0].split('"')[1].split(':')]
    relation = output_parts[0].strip()
    words = [word.strip() for word in output_parts[1].split(',')]
    
    print(f"Task {idx + 1}: {task_name}")
    print("Output:")
    for word in words:
        print(f"- {word}")
    print()
