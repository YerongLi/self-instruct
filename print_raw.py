import json

# Read the JSONL file
with open('data/ie/70machine_generated_instances.jsonl', 'r') as file:
    lines = file.readlines()

# Create a file to write the output
with open('text.txt', 'w') as output_file:
    # Parse JSONL and write truncated raw_instances to the file
    for line in lines:
        data = json.loads(line)
        raw_instances = data.get('raw_instances')
        if raw_instances:
            # Find the second occurrence of "Task:"
            index = raw_instances.find("Task:", raw_instances.find("Task:") + 1)
            if index != -1:
                truncated_raw_instances = raw_instances[:index]
            else:
                truncated_raw_instances = raw_instances

            output_file.write(truncated_raw_instances + "\n")
            output_file.write("=" * 80 + "\n")