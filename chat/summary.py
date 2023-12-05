import os

directory_path = "sampled"  # Change this to the desired directory path

# Check if the directory exists
prompts = []
prefix = "Could you give a summary on the following dialogue, keeping [PERSON], [LOC] as privacy tokens?\n"
suffix = "\nSummary:\n"
if os.path.exists(directory_path) and os.path.isdir(directory_path):
    # Walk through the directory and its immediate subdirectories (depth=1)
    for root, dirs, files in os.walk(directory_path):
        # Ignore subdirectories beyond depth=2
        if root[len(directory_path):].count(os.sep) <= 2:
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as txt_file:
                        file_contents = txt_file.read()
                        prompts.append({'prompt': prefix + file_contents + suffix, 'filename' : file_path})
                        # print(f"Contents of {file_path}:\n{file_contents}\n{'-' * 50}")
else:
    print(f"The directory '{directory_path}' does not exist.")
