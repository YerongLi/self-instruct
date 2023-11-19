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
# Function to process a key and update the shared dictionary
def process_key(key):
    value = data[key]
    result = {value['wd_node']: value['wd_description']} if 'wd_description' is data[key] else {}
    
    if 'overlay_parents' in value:
        overlay_parents = value['overlay_parents']
        for overlay_parent in overlay_parents:
            if 'wd_node' in overlay_parent and overlay_parent['wd_node'] not in result:
                result[overlay_parent['wd_node']] = get_wikidata_description(overlay_parent['wd_node'])

    return result
keys = sorted(data.keys(), key=lambda x: x[::-1])
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

pool = multiprocessing.Pool()

# Use the pool to process keys in parallel
results = list(tqdm(pool.imap(process_key, keys), total=len(keys)))

# Close the pool to release resources
pool.close()
pool.join()

ans = {}
for result in results:
    ans.update(result)

# Randomly print 5 entries from the dictionary
random_entries = random.sample(list(ans.keys()), 5)
for key in random_entries:
    print(key, ans[key])

# Dump the dictionary to a JSON file
with open("dictionary.json", "w") as json_file:
    json.dump(ans, json_file, indent=2)

# Count None values and total items
none_count = sum(1 for value in ans.values() if value is None)
print(f"Number of 'None' values: {none_count}")
print(f"Total items: {len(ans)}")
print(f"Percentage of 'None' values: {(none_count / len(ans)) * 100:.2f}%")