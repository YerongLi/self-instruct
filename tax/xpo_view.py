import json

# Load the JSON file
with open('xpo_v1.1.json', 'r') as json_file:
    data = json.load(json_file)

# Iterate through the data and print the "name" values under "overlay_parents"
for key, value in data.items():
    print(value['name'])
    if 'overlay_parents' in value:
        overlay_parents = value['overlay_parents']
        for overlay_parent in overlay_parents:
            # if 'name' in overlay_parent:
                print(f"   Overlay Parent name: {overlay_parent['name']}")
    print()  # Print a newline after each entry


print(len(data))