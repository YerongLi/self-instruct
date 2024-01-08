import json

def split_and_count(input_file):
    police1_data = []
    police_data = []

    type_set = ['DrugsAlcohol', 'HarassmentAbuse', 'MentalHealth', 'TheftLostItem', 'SuspiciousActivity', 'EmergencyMessage', 'AccidentTrafficParking', 'NoiseDisturbance', 'FacilitiesMaintenance']

    with open(input_file, 'r') as infile:
        for line in infile:
            record = json.loads(line)
            
            if len(police_data) % 4 == 0:
                police1_data.append(record)
            else:
                police_data.append(record)

            # Count the number of each type
            # record_type = record.get("type")
            # type_counts[record_type] += 1

    with open('police.json', 'w') as outfile:
        for record in police_data:
            json.dump(record, outfile)
            outfile.write('\n')

    with open('police1.json', 'w') as outfile1:
        for record in police1_data:
            json.dump(record, outfile1)
            outfile1.write('\n')
    type_counts = {record_type: 0 for record_type in type_set}
    print("\nNumber of each type in police1.json:")
    for record in police_data:
        record_type = record.get("type")
        type_counts[record_type] += 1

    for record_type in type_set:
        print(f"{record_type}: {type_counts[record_type]}")
    print("Number of each type in police.json:")
    for record_type in type_set:
        print(f"{record_type}: {type_counts[record_type]}")

    type_counts = {record_type: 0 for record_type in type_set}
    print("\nNumber of each type in police1.json:")
    for record in police1_data:
        record_type = record.get("type")
        type_counts[record_type] += 1

    for record_type in type_set:
        print(f"{record_type}: {type_counts[record_type]}")

if __name__ == "__main__":
    input_file = "police-full.json"  # Replace with the actual file path
    split_and_count(input_file)
