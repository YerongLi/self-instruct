import requests

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

# Example usage:
wikidata_id = "Q7239"
description = get_wikidata_description(wikidata_id)

if description:
    print(f"The description for {wikidata_id} is: {description}")
else:
    print("Unable to retrieve the description.")

