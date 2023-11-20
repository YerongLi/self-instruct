import os
import re

def find_text_in_files(folder_path):
    unique_texts = set()

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)

                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    matches = re.findall(r'\[([^\]]+)\]', content)

                    for match in matches:
                        unique_texts.add(match)

    for text in unique_texts:
        print(f"[{text}]")

    return unique_texts

# Example: Replace 'your_folder_path' with the actual path to your folder
folder_path = 'sampled'
unique_texts = find_text_in_files(folder_path)

