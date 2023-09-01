import json

# Define the rewrite function
def rewrite(schema):
    return f"rewrite {schema}"

# Read the JSONL file
with open('data/seed_task_ie.jsonl', 'r') as file:
    lines = file.readlines()

# Create a list to store rewritten tasks
rewritten_tasks = []

# Process each line in the JSONL file
for line in lines:
    data = json.loads(line)
    
    if 'schema' in data:
        instruction = data['instruction']
        schema = data['schema'].replace('Text: {0}\nAnswer:', '')
        input_text = data['input']
        output_text = data['output']
        
        # Use the rewrite function to add the "rewrite" prefix to schema
        rewritten_schema = rewrite(schema)
        
        # Create a dictionary for the rewritten task
        rewritten_task = {
            'instruction': instruction,
            'schema': rewritten_schema,
            'input': input_text,
            'output': output_text
        }
        
        # Append the rewritten task to the list
        rewritten_tasks.append(rewritten_task)

# Write all rewritten tasks to a new file
with open('data/rewrite_seed_task_ie.jsonl', 'w') as rewrite_file:
    for task in rewritten_tasks:
        rewrite_file.write(json.dumps(task) + '\n')

print('All tasks have been rewritten and stored in "data/rewrite_seed_task_ie.jsonl"')
