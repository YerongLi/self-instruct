import json

# Read the JSONL file
data_file = 'data/seed_task_ie.jsonl'

# Create a dictionary to store instructions
instructions_dict = {}

# Open and read the JSONL file
with open(data_file, 'r') as f:
    for line in f:
        # Parse the JSON line
        data = json.loads(line)
        
        # Extract instruction and name
        instruction = data['instruction']
        name = data['name']
        
        # Add instruction to the set in the dictionary
        if name not in instructions_dict:
            instructions_dict[name] = set()
        instructions_dict[name].add(instruction)

# Print the dictionary with names as keys and sets of instructions as values
for name, instruction_set in instructions_dict.items():
    print(f"Name: {name}")
    print(f"Instructions: {instruction_set}\n")
