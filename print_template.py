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
    
    # Find the schema part in the instruction
    schema_start_idx = schema.find('Output format should be "') + len('Output format should be "')
    schema_end_idx = schema.find('"', schema_start_idx)
    
    if schema_start_idx != -1 and schema_end_idx != -1:
        schema_part = schema[schema_start_idx:schema_end_idx]
        relations_and_words = [item.strip() for item in schema_part.split(';')]
        
        print(f"Task {idx + 1}: {task_name}")
        print("Output:")
        for relation_and_words in relations_and_words:
            relation, words = [item.strip() for item in relation_and_words.split(':')]
            word_list = [word.strip() for word in words.split(',')]
            for word in word_list:
                print(f"- {word} (Relation: {relation})")
        print()
