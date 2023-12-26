import json
import os
import pickle
from tqdm import tqdm
import pandas as pd


# Specify the correct encoding of your CSV files
encoding = 'latin-1'  # or 'ISO-8859-1' or other suitable encoding
filename = 'police-full.json'
result_type_set = set()
if os.path.exists(filename):
    # Remove the file
    os.remove(filename)
# Read the event CSV file to create a hashmap from event ID to event category
print('Creating df_event.csv file')
event_df = pd.read_pickle('df_event.pkl')
event_type_map = {}
del event_df['Event Text']

type_set = {'DrugsAlcohol', 'HarassmentAbuse', 'MentalHealth', 'TheftLostItem', 'SuspiciousActivity', 'EmergencyMessage', 'SafeRide&SafeWalk', 'NoiseDisturbance', 'FacilitiesMaintenance'}

# Iterate over rows in the event_df DataFrame
for index, row in tqdm(event_df.iterrows(), total=event_df.shape[0]):
    event_id = row['Anonymized Event ID']
    # print(event_id)
    if index % 2 == 0: continue
    # event_category = row['Eventcategory']
    event_category = row['Eventcategory (group)'].replace('/', '').replace(' ', '')
    if event_category not in type_set:
        print(event_category)
    if event_category == 'unknown' or event_category not in type_set: continue

    # if event_id == 2073482:
        # print(event_category)

    event_type_map[event_id] = event_category
# Dump event_df to event_df.csv
event_df.to_csv('event_df.csv', index=False)


print('Creating chat_df.csv file')
chat_df = pd.read_pickle('df_chat.pkl')
chat_df['Chat Date'] = pd.to_datetime(chat_df['Chat Date'])

chat_df.sort_values(by=['Anonymized Eventid', 'Chat Date'], inplace=True)
unique_chat_types = set(chat_df['Chattype'])
print(unique_chat_types)


# Dump chat_df to chat_df.csv
chat_df.to_csv('chat_df.csv', index=False)

# Create a dictionary to store the result
result_dict = {}
not_good = set()
# for index, row in tqdm(chat_df.iterrows(),total=chat_df.shape[0]):

#     event_id = row['Anonymized Eventid']
#     event_type = event_type_map.get(event_id, 'unknown')  # Get event category from the hashmap
#     chat_history = row['Chat']
#     if event_id in not_good: continue
#     if event_type == 'unknown' or event_type not in type_set: continue
#     chat_history = row['Chat']
#     if not isinstance(chat_history, str):
#         not_good.add(event_id)

#     # Check if the event_id is already in the dictionary
#     if event_id in result_dict:
#         # Append the chat history to the existing list
#         result_dict[event_id]['chat'].append(chat_history)
#     else:
#         # Create a new dictionary entry for the event_id
#         result_dict[event_id] = {'chat': [chat_history]}
#     result_dict[event_id]['his_len'] = len(result_dict[event_id]['chat'])
#     # if len(result_dict[event_id]['chat']) % 2 == 0:
# for event_id in result_dict:
#     del result_dict[event_id]['chat']
#     result_dict[event_id]['chat'] = []
# Iterate through rows in the chat dataframe
count = 0
max_len = 0
previous_event_id = None
for index, row in tqdm(chat_df.iterrows(),total=chat_df.shape[0]):
    event_id = row['Anonymized Eventid']
    if event_id in not_good: continue # No error
    event_type = event_type_map.get(event_id, 'unknown')  # Get event category from the hashmap
    if event_type == 'unknown' or event_type not in type_set: continue
    result_type_set.add(event_type)
    chat_turn = row['Chat']
    chat_type = row['Chattype']
    if chat_type not in {"Admin", "User"}: continue 
    if event_id == previous_event_id and chat_type != chat_history[-1][1]:
        chat_history.append((chat_turn, chat_type))  # Append to existing chat_history
    else:
        chat_history = [(chat_turn, chat_type)]  # Start a new chat_history
    # print([item[1] for item in chat_history])
    previous_event_id = event_id  # Update previous_event_id for the next iteration
    if len(chat_history) % 2 == 0 and len(chat_history) <= 70 and chat_history[-1][1] == 'Admin':
        entry = {
            'history': [[chat_history[i][0], chat_history[i+1][0]] for i in range(0, len(chat_history) - 2, 2)],  # Concatenate pairs
            'instruction': str(chat_history[-2][0]),
            'output': str(chat_history[-1][0])
        }
        with open(filename, 'a') as json_file:
          json.dump(entry, json_file)
          json_file.write('\n') # Add a newline for better readability
          count+= 1
print(count)
print(result_type_set)
print(max_len)