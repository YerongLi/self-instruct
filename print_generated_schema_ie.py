import json

# Define the path to the JSONL file
file_path = 'data/ie/machine_generated_instructions.jsonl'

# Function to read and print "instruction" fields from the JSONL file
def print_instructions_from_jsonl(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                instruction = data.get("raw_instances")
                if instruction:
                    print(instruction)
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Call the function to print instructions
print_instructions_from_jsonl(file_path)
