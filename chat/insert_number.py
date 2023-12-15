import json
import os

# Define the file path
file_path = 'police1.json'  # Replace with the actual file path

# Read JSON file line by line
with open(file_path, 'r') as file:
    for line in file:
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

                # Print the counts for the current entry
                # print(f"File: {filename}")
                # print(f"User lines count: {user_count}")
                # print(f"Admin lines count: {admin_count}")
                # print("\n")
            else:
                print(f"File not found: {filename}")

        except json.JSONDecodeError:
            print("Error decoding JSON. Check the format of the JSON lines.")
