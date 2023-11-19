import json
import multiprocessing
import random
import requests
import sys
from tqdm import tqdm
# Load the JSON file
with open('xpo_v1.1.json', 'r') as json_file:
    data = json.load(json_file)

# Iterate through the data and print the "name" values under "overlay_parents"

# Assuming 'data' is a dictionary containing the data
keys = sorted(data.keys(), key=lambda x: x[::-1])
start_index = -1
# Iterate over the keys and print the iteration number and corresponding values
def get_wikidata_description(wikidata_id):
    # Wikidata API endpoint
    api_endpoint = "https://www.wikidata.org/w/api.php"

    # Parameters for the API request
    params = {
        'action': 'wbgetentities',
        'ids': wikidata_id,
        'format': 'json',
        'props': 'descriptions',
        'languages': 'en',  # Change 'en' to the desired language code
    }

    try:
        # Make the API request
        response = requests.get(api_endpoint, params=params)
        response.raise_for_status()  # Check for errors in the HTTP response

        # Parse the JSON response
        data = response.json()

        # Get the description from the response
        description = data['entities'][wikidata_id]['descriptions']['en']['value']

        return description

    except Exception as e:
        print(wikidata_id)
        print(f"Error making the request: {e}")
        return None

def process_keys(start_index, keys, shared_dict):
    for i, key in tqdm(enumerate(keys), total=len(keys)):
        if i <= start_index:
            continue
        value = data[key]
        shared_dict[data[key]['wd_node']] = data[key]['wd_description'] if 'wd_description' in data[key] else None
        if 'overlay_parents' in value:
            overlay_parents = value['overlay_parents']
            for overlay_parent in overlay_parents:
                if 'wd_node' in overlay_parent and overlay_parent['wd_node'] not in shared_dict:
                    shared_dict[overlay_parent['wd_node']] = get_wikidata_description(overlay_parent['wd_node'])

# Assuming 'data', 'keys', and 'start_index' are defined before this point

# Create a Manager to manage the shared dictionary
manager = multiprocessing.Manager()
shared_dict = manager.dict()

# Set the number of processes (adjust as needed)
num_processes = multiprocessing.cpu_count()

# Divide the keys among processes
chunk_size = len(keys) // num_processes
processes = []

for i in range(num_processes):
    start = i * chunk_size
    end = (i + 1) * chunk_size if i < num_processes - 1 else len(keys)
    process = multiprocessing.Process(target=process_keys, args=(start, keys[start:end], shared_dict))
    processes.append(process)

# Start the processes
for process in processes:
    process.start()

# Wait for all processes to finish
for process in processes:
    process.join()

# Dump the shared dictionary to a JSON file
with open("dictionary.json", "w") as json_file:
    json.dump(dict(shared_dict), json_file, indent=2)
none_count = 0
for key in shared_dict:
    if shared_dict[key] == None:
        none_count+= 1
print(f"{none_count} / {len(shared_dict)}")