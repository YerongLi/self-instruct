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
    schema_start_idx = schema.find('Output format is "') + len('Output format is "')
    schema_end_idx = schema.find('"', schema_start_idx)
    
    if schema_start_idx != -1 and schema_end_idx != -1:
        schema_part = schema[schema_start_idx:schema_end_idx]
        options_start_idx = schema_part.find('Option: ') + len('Option: ')
        options_end_idx = schema_part.find('\n', options_start_idx)
        options = schema_part[options_start_idx:options_end_idx].split(', ')
        
        text_start_idx = schema_part.find('Text: {0}') + len('Text: {0}\n')
        text_end_idx = schema_part.find('\n', text_start_idx)
        text = schema_part[text_start_idx:text_end_idx]
        
        output_start_idx = schema.find('output": "') + len('output": "')
        output_end_idx = schema.find('", "is_classification":', output_start_idx)
        output = schema[output_start_idx:output_end_idx]
        
        print(f"Task {idx + 1}: {task_name}")
        print("Text:", text)
        print("Output:")
        
        for option in options:
            option_parts = option.split(': ')
            if len(option_parts) > 1:
                option_type = option_parts[0]
                option_words = [word.strip() for word in option_parts[1].split(', ')]
                for word in option_words:
                    print(f"- {word} ({option_type})")
        print("Answer:", output)
        print()
