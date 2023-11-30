import argparse
import json

def view_nested_json(filename):
    # Load the JSON file
    with open(filename, 'r') as file:
        predictions = json.load(file)

    # Print the nested structure with line breaks
    formatted_predictions = json.dumps(predictions, indent=2).replace("\\n", "\n")
    print(formatted_predictions)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="View the contents of a depth-2 nested JSON file.")
    parser.add_argument("filename", help="Path to the JSON file")

    # Parse command-line arguments
    args = parser.parse_args()

    # Call the function with the provided filename
    view_nested_json(args.filename)
