import json
import argparse

parser = argparse.ArgumentParser(description='Process some files.')

# Add the role argument
parser.add_argument('role', type=str, help='Role to construct the filename')

# Parse the arguments
args = parser.parse_args()
def split_and_count(input_file):
    police1_data = []
    police_data = []

    type_set = ['SuspiciousActivity', 'AccidentTrafficParking', 'DrugsAlcohol', 'EmergencyMessage', 'FacilitiesMaintenance', 'HarassmentAbuse', 'MentalHealth', 'NoiseDisturbance', 'TheftLostItem']


    with open(input_file, 'r') as infile:
        for i, line in enumerate(infile):
            record = json.loads(line)
            
            if i % 4 == 0:
                police1_data.append(record)
            else:
                police_data.append(record)

    with open(f'{args.role}_train.jsonl', 'w') as outfile:
        for record in police_data:
            json.dump(record, outfile)
            outfile.write('\n')

    with open(f'{args.role}_test.jsonl', 'w') as outfile1:
        for record in police1_data:
            json.dump(record, outfile1)
            outfile1.write('\n')

    type_counts_police = {record_type: 0 for record_type in type_set}
    print("\nNumber of each type in police.jsonl:")
    for record in police_data:
        record_type = record.get("type")
        type_counts_police[record_type] += 1

    total_entries_police = sum(type_counts_police.values())
    print(f"Total entries in police.jsonl: {total_entries_police}")
    for record_type in type_set:
        print(f"{record_type}: {type_counts_police[record_type]}")

    type_counts_police1 = {record_type: 0 for record_type in type_set}
    print("\nNumber of each type in police1.jsonl:")
    for record in police1_data:
        record_type = record.get("type")
        type_counts_police1[record_type] += 1

    total_entries_police1 = sum(type_counts_police1.values())
    print(f"Total entries in police1.json: {total_entries_police1}")
    for record_type in type_set:
        print(f"{record_type}: {type_counts_police1[record_type]}")

if __name__ == "__main__":
    input_file = f"{args.role}-full.jsonl"  # Replace with the actual file path
    split_and_count(input_file)
