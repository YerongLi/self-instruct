import json
import os
import pickle
from tqdm import tqdm
import pandas as pd


# Specify the correct encoding of your CSV files
encoding = 'latin-1'  # or 'ISO-8859-1' or other suitable encoding
filename = 'police-full.json'
if os.path.exists(filename):
    # Remove the file
    os.remove(filename)
# Read the event CSV file to create a hashmap from event ID to event category
# event_df = pd.read_csv('event.csv', encoding=encoding)
# event_df = pickle.load(open('df_event.pkl', 'rb'))
event_df = pd.read_pickle('df_event.pkl')
event_type_map = {}

# Iterate over rows in the event_df DataFrame
for index, row in tqdm(event_df.iterrows(), total=event_df.shape[0]):
    event_id = row['Anonymized Event ID']
    # print(event_id)
    if index % 2 == 0: continue
    # event_category = row['Eventcategory']
    event_category = row['Eventcategory (group)'].replace('/', '').replace(' ', '')
    # if event_id == 2073482:
        # print(event_category)

    # if index != 4413 and index != 4414 : continue
    # print('id')
    # print(event_id)
    if len(event_category) > 40 and event_category not in {'DrugsAlcohol', 'HarassmentAbuse', 'MentalHealth', 'TheftLostItem', 'SuspiciousActivity', 'EmergencyMessage'}:
        print(event_category)
        print('==========')
    event_type_map[event_id] = event_category
# print(event_type_map[1992077])
# Read the chat CSV file containing chat data
# chat_df = pd.read_csv('chat.csv', encoding=encoding)
chat_df = pd.read_pickle('df_chat.pkl')

# Create a dictionary to store the result
result_dict = {}

for index, row in tqdm(chat_df.iterrows(),total=chat_df.shape[0]):

    event_id = row['Anonymized Eventid']
    event_type = event_type_map.get(event_id, 'unknown')  # Get event category from the hashmap
    if event_type == 'unknown': continue
    chat_history = row['Chat']

    # Check if the event_id is already in the dictionary
    if event_id in result_dict:
        # Append the chat history to the existing list
        result_dict[event_id]['chat'].append(chat_history)
    else:
        # Create a new dictionary entry for the event_id
        result_dict[event_id] = {'chat': [chat_history]}
    result_dict[event_id]['his_len'] = len(result_dict[event_id]['chat'])
    if len(result_dict[event_id]['chat']) % 2 == 0:
for event_id in result_dict:
    del result_dict[event_id]['chat']
    result_dict[event_id]['chat'] = []
# Iterate through rows in the chat dataframe
for index, row in tqdm(chat_df.iterrows(),total=chat_df.shape[0]):
    event_id = row['Anonymized Eventid']
    event_type = event_type_map.get(event_id, 'unknown')  # Get event category from the hashmap
    if event_type == 'unknown': continue
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
        history = [['', result_dict[event_id]['chat'][0]]]
        # history.extend(
            # [result_dict[event_id]['chat'][i:i+2] for i in range(1, len(result_dict[event_id]['chat']), 2)])
        event_type = event_type_map.get(event_id, 'unknown')  # Get event category from the hashmap

        # Dump the entry to police.json
        entry = {
            "instruction": instruction,
            "history": history,
            "type": event_type,
            "output": chat_history,
            'his_len': result_dict[event_id]['his_len'],
            # Add other fields as needed
        }

        with open(filename, 'a') as json_file:
            json.dump(entry, json_file)
            json_file.write('\n')  # Add a newline for better readability

# Print or store the result_dict as needed
# for event_id, data in result_dict.items():
    # print(f"Event {event_id}: {data}")
print(count)