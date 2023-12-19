import pandas as pd

# Read the CSV file
df = pd.read_csv('chat.csv', encoding='latin-1')

# Initialize a dictionary to store conversations for each event
event_conversations = {}

# Iterate through rows in the dataframe
for index, row in df.iterrows():
    event_id = row['Anonymized Eventid']
    chat_history = row['Chat']

    # Check if the event_id is already in the dictionary
    if event_id in event_conversations:
        # Append the chat history to the existing list
        event_conversations[event_id].append(chat_history)
    else:
        # Create a new list for the event_id and add the chat history
        event_conversations[event_id] = [chat_history]
# Print or store the result as needed
print(len(event_conversations))
# for event_id, conversations in event_conversations.items():
#     print(f"Event {event_id}: {conversations}")
