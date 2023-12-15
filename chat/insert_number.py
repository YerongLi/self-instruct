import json
import os

# Define the file path
input_file_path = 'police1.json'  # Replace with the actual file path
output_file_path = 'police2.json'  # Replace with the actual file path
modified_entries = []

# Read JSON file line by line
with open(input_file_path, 'r') as input_file:
    for line in input_file:
        try:
            # Parse each line as JSON
            entry = json.loads(line)

            # Extract the "filename" key
            filename = entry.get("filename", "")

            if os.path.isfile(filename):
                user_count = admin_count = 0

                # Read the content of the file
                with open(filename, 'r') as entry_file:
                    lines = entry_file.readlines()

                    # Count lines starting with "User" or "Admin"
                    for line in lines:
                        if line.startswith("User"):
                            user_count += 1
                        elif line.startswith("Admin"):
                            admin_count += 1

                # Add "his_len" field to the entry
                entry['his_len'] = user_count + admin_count

                # Append the modified entry to the list
                modified_entries.append(entry)
            else:
                print(f"File not found: {filename}")

        except json.JSONDecodeError:
            print("Error decoding JSON. Check the format of the JSON lines.")

# Dump the modified entries into a new JSON file
with open(output_file_path, 'w') as output_file:
    for modified_entry in modified_entries:
        json.dump(modified_entry, output_file)
        output_file.write('\n')