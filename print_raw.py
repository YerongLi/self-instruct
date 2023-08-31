import json

# Read the JSONL file
with open('data/ie/machine_generated_instances.jsonl', 'r') as file:
    lines = file.readlines()

# Parse JSONL and print raw_instances
for line in lines:
    data = json.loads(line)
    raw_instances = data.get('raw_instances')
    if raw_instances:
        print(raw_instances)
        print("=" * 80)  # Separator