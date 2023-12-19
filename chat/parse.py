import json
import pandas as pd

# Specify the correct encoding of your CSV files
encoding = 'latin-1'  # or 'ISO-8859-1' or other suitable encoding

# Read the first CSV file containing chat data
chat_df = pd.read_csv('chat.csv', encoding=encoding)

# Read the second CSV file containing event details
event_df = pd.read_csv('event.csv', encoding=encoding)

# Create a dictionary to store the result
result_dict = {}

# Iterate through rows in the chat dataframe
for index, row in chat_df.iterrows():
    event_id = row['Anonymized Eventid']
    chat_history = row['Chat']

    # Check if the event_id is already in the dictionary
    if event_id in result_dict:
        # Append the chat history to the existing list
        result_dict[event_id]['chat'].append(chat_history)
    else:
        # Create a new dictionary entry for the event_id
        result_dict[event_id] = {'chat': [chat_history]}
current_event_type = None
# Iterate through rows in the event dataframe to add event category
for index, row in event_df.iterrows():
    event_id = row['Anonymized Event ID']
    event_category = row['Eventcategory']

    # Check if the event_id is in the result_dict
    if event_id in result_dict:
        # Add the event category to the dictionary entry
        result_dict[event_id]['type'] = event_category
for index, row in chat_df.iterrows():
    event_id = row['Anonymized Eventid']
    chat_history = row['Chat']

    # Check if the event_id is already in the dictionary
    if event_id in result_dict:
        # Append the chat history to the existing list
        result_dict[event_id]['chat'].append(chat_history)
    else:
        # Create a new dictionary entry for the event_id
        result_dict[event_id] = {'chat': [chat_history]}

    # Check if the length of chat_history is even
    if len(result_dict[event_id]['chat']) % 2 == 0:
        # Extract information for the police.json entry
        instruction = result_dict[event_id]['chat'][-1]
        history = result_dict[event_id]['chat'][:-1]
        event_type = current_event_type  # Use the previously stored event type

        # Dump the entry to police.json
        entry = {
            "instruction": instruction,
            "history": history,
            "type": event_type,
            # Add other fields as needed
        }
        with open('police.json', 'a') as json_file:
            json.dump(entry, json_file)
            json_file.write('\n')  # Add a newline for better readability

# Print or store the result_dict as needed
for event_id, data in result_dict.items():
    print(f"Event {event_id}: {data}")

