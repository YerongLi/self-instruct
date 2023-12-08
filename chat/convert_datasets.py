from datetime import date
import json
import argparse
import os


HISTORY_LIMIT = 10
def extract_conversation(filename, name, json_filename, summary_json_filename):
    # Open the input file
    with open(filename, "r") as file:
        content = file.readlines()
    with open(summary_json_filename, 'r') as json_file:
        summary = json.load(json_file)
    # Extract the conversation between the user and the specified name
    conversation = []
    for line in content:
    #     if f"[{name}]" in line:
    #         role = "Assistant"
    #     else:
    #         role = "Human"
    #     utterance = line[line.index("]") + 1:].strip()
        # if f"User" in line:
        if f"Admin" in line:
            role = "Assistant"
        else :
            role = "Human"
        utterance = line.split(" ", 1)[1].strip().strip('"')
        conversation.append({"role": role, "utterance": utterance})

    # Prepare the data in JSON format
    data = {
        "instruction": None,
        "output": None,
        "history": [],
        "summary": None,
        "filename": None,
    }
    for i in range(len(conversation) - 1):
        if not (conversation[i]["role"] == "Human" and conversation[i + 1]["role"] == "Assistant"):
            continue
        history_start_idx = max(0, i - HISTORY_LIMIT)
        history = [conv["utterance"] for conv in conversation[history_start_idx:i]]
        if len(history) % 2 != 0:
            continue
        history = [[history[i], history[i + 1]] for i in range(0, len(history), 2) if i + 1 < len(history)]
        instruction = conversation[i]["utterance"]
        output = conversation[i + 1]["utterance"]
        data["instruction"] = instruction
        data["output"] = output
        data["history"] = history
        data["summary"] = summary[filename]['o']
        data["filename"] = filename
        data["type"] = filename.split('/')[-2]
        
        # Dump each conversation as a separate JSON object in the file
        with open(json_filename, "a") as f:
            json.dump(data, f)
            f.write("\n")

def extract_single_conversation(filename, name, json_filename):
    # Open the input file
    with open(filename, "r") as file:
        content = file.readlines()

    # Extract the conversation between the user and the specified name
    conversation = []
    for line in content:
        # if f"[{name}]" in line:
        if f"User" in line:
            role = "Assistant"
        else :
            role = "Human"
        utterance = utterance = line.split(" ", 1)[1].strip().strip('"')
        conversation.append({"role": role, "utterance": utterance})

    # Prepare the data in JSON format
    data = {
        "instruction": None,
        "output": None,
        "history": []
    }
    for i in range(len(conversation) - 1):
        if not (conversation[i]["role"] == "Human" and conversation[i + 1]["role"] == "Assistant"):
            continue
        instruction = conversation[i]["utterance"]
        output = conversation[i + 1]["utterance"]
        data["instruction"] = instruction
        data["output"] = output
        # Dump each conversation as a separate JSON object in the file
        with open(json_filename, "a") as f:
            json.dump(data, f)
            f.write("\n")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process conversation files")
    parser.add_argument("--dir", help="Directory to search for '.txt' files recursively", required=True)
    parser.add_argument("--su", help="summary json file", default='sampled/summary.json')
    parser.add_argument("--name", help="Name to replace (default: 'police')", default="police")
    args = parser.parse_args()
    # Define the JSON file name
    name = args.name
    json_filename = f"{name.lower().replace(' ', '_')}.json"
    # Process "clean_" files
    for root, dirs, files in os.walk(args.dir):
        if root[len(args.dir):].count(os.sep) > 1:
            continue  # Skip files beyond depth 2
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                extract_conversation(file_path, name, json_filename, args.su)

    # Process "single_turn.txt" if it exists
    single_turn_filename = "single_turn.txt"
    # if os.path.exists(single_turn_filename):
        # extract_single_conversation(single_turn_filename, name, json_filename)
if __name__ == "__main__":
    main()