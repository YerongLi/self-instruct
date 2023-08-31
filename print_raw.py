# Read the JSONL file
with open('data/ie/machine_generated_instances.jsonl', 'r') as file:
    lines = file.read().split('\n')

# Separate instances and print
current_instance = ""
for line in lines:
    if line.strip() != "":
        current_instance += line + "\n"
    else:
        if current_instance.strip() != "":
            print(current_instance.strip())
            print("\n" + "=" * 80 + "\n")
            current_instance = ""

# Print the last instance if there's any left
if current_instance.strip() != "":
    print(current_instance.strip())
