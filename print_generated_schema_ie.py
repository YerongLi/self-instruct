import json
import re
def extract_prefix_until_index(input_string):
    lines = input_string.split('\n')  # Split the string into lines

    output = []

    for line in lines:
        # Use regular expression to check if the line starts with an index
        if re.match(r'^\d+\.\s', line):
            break
        output.append(line)

    result = '\n'.join(output)
    return result

# Define the path to the JSONL file
file_path = 'data/ie/machine_generated_format.jsonl'

# Function to read and print "instruction" fields from the JSONL file
def print_instructions_from_jsonl(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                instruction = data.get("instruction")
                raw_format = data.get("raw_instances")
                clean_format = extract_prefix_until_index(raw_format)
                if instruction:
                    print(f"Instruction\n {instruction.rstrip()}")
                    print(f"Format\n {clean_format}")
                    print("===")
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Call the function to print instructions
print_instructions_from_jsonl(file_path)
