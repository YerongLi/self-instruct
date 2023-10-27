import json
import sys
# Load the JSON file
with open('xpo_v1.1.json', 'r') as json_file:
    data = json.load(json_file)

# Iterate through the data and print the "name" values under "overlay_parents"

# Assuming 'data' is a dictionary containing the data
keys = sorted(data.keys(), key=lambda x: x[::-1])
start = 30
# Iterate over the keys and print the iteration number and corresponding values
for i, key in enumerate(keys):
    if i <= 40 : continue
    value = data[key]
    print(f"Iteration {i+1}\n Name: {value['name']}")

    if 'overlay_parents' in value:
        overlay_parents = value['overlay_parents']
        for overlay_parent in overlay_parents:
            if 'name' in overlay_parent:
                print(f"   Overlay Parent name: {overlay_parent['name']}")

    print()  # Print a newline after each entry

    # Check if it's time to pause and wait for the enter key
    if (i + 1) % 5 == 0:
        sys.stdin.readline()  # Wait for the enter key