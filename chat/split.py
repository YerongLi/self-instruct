import json

def split_and_count(input_file):
    with open(input_file, 'r') as infile:
        data = json.load(infile)

    police1_data = []
    police_data = []

    for idx, record in enumerate(data):
        if (idx + 1) % 4 == 0:
            police1_data.append(record)
        else:
            police_data.append(record)

    with open('police.json', 'w') as outfile:
        json.dump(police_data, outfile)

    with open('police1.json', 'w') as outfile1:
        json.dump(police1_data, outfile1)

    # Count the number of each type in the specified order
    type_set = ['DrugsAlcohol', 'HarassmentAbuse', 'MentalHealth', 'TheftLostItem', 'SuspiciousActivity', 'EmergencyMessage', 'AccidentTrafficParking', 'NoiseDisturbance', 'FacilitiesMaintenance']
    type_counts = {record_type: 0 for record_type in type_set}

    for record in data:
        record_type = record.get("type")
        type_counts[record_type] += 1

    print("Number of each type:")
    for record_type in type_set:
        print(f"{record_type}: {type_counts[record_type]}")

if __name__ == "__main__":
    input_file = "police-full.json"
    split_and_count(input_file)
