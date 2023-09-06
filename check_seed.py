# Read the JSONL file
try:
    with open('data/seed_task_ie.jsonl', 'r') as file:
        lines = file.readlines()
    print("The file 'data/seed_task_ie.jsonl' is readable.")
except FileNotFoundError:
    print("The file 'data/seed_task_ie.jsonl' does not exist.")
except IOError as e:
    print("An error occurred while reading the file 'data/seed_task_ie.jsonl'.")
    print("Error message:", str(e))