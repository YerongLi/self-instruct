import os

directory_path = "sampled"  # Change this to the desired directory path

# Check if the directory exists
if os.path.exists(directory_path) and os.path.isdir(directory_path):
    # Get a list of all *.txt files in the directory
    txt_files = [file for file in os.listdir(directory_path) if file.endswith(".txt")]
    print(txt_files)
    # Read and print the contents of each *.txt file
    for txt_file in txt_files:
        file_path = os.path.join(directory_path, txt_file)
        with open(file_path, 'r') as file:
            file_contents = file.read()
            print(f"Contents of {txt_file}:\n{file_contents}\n{'-' * 50}")
else:
    print(f"The directory '{directory_path}' does not exist.")
