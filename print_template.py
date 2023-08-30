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
    input_text = task['input']
    schema = task['schema']
    
    output_start_idx = schema.find('output": "') + len('output": "')
    output_end_idx = schema.find('", "is_classification":', output_start_idx)
    output = schema[output_start_idx:output_end_idx]
    
    print(f"Task {idx + 1}: {task_name}")
    print("Input text:", input_text)
    print("Output:")
    
    relation_word_pairs = output.split('; ')
    for relation_word_pair in relation_word_pairs:
        relation, words = relation_word_pair.split(': ')
        word_list = [word.strip() for word in words.split(', ')]
        for word in word_list:
            print(f"- {word} (Relation: {relation})")
    print()
