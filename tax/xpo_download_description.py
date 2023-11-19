import json
import requests
import sys
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

    except requests.exceptions.RequestException as e:
        print(f"Error making the request: {e}")
        return None

ans = {}
for i, key in enumerate(keys):
    if i <= start_index : continue
    value = data[key]
    print(f"Iteration {i+1}\n Name: {value['name']}")

    if 'overlay_parents' in value:
        overlay_parents = value['overlay_parents']
        for overlay_parent in overlay_parents:
            # print(overlay_parents)
            if 'wd_node' in overlay_parent and overlay_parent['wd_node'] not in ans:
                ans[overlay_parent['wd_node']] = get_wikidata_description(overlay_parent['wd_node'])
                # print(f"   Overlay Parent name: {overlay_parent['name']}")

    print()  # Print a newline after each entry
random_entries = random.sample(list(ans.keys()), 5)

for key in random_entries:
    print(key, ans[key])
