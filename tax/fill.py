error_dict = {}  # Initialize the dictionary to store pairs of current_name and current_parent

# Assuming 'error.txt' is the name of your file
with open('error.txt', 'r') as file:
    for line in file:
        line = line.strip()  # Remove leading and trailing whitespaces
        if line.startswith('Iteration'):
            # Skip lines starting with "Iteration"
            continue
        elif 'Name:' in line:
            # Extract the name from lines containing "Name:"
            current_name = line.split('Name:', 1)[1].strip()
            print(f"Current Name: {current_name}")
        elif 'Overlay Parent name:' in line:
            # Extract the parent from lines containing "Overlay Parent name:"
            current_parent = line.split('Overlay Parent name:', 1)[1].strip()
            print(f"Current Parent: {current_parent}")
            print() 
            # Save the pair in the dictionary
            error_dict[current_name] = current_parent

# Now, 'error_dict' contains pairs of current_name and current_parent
print("Final Error Dictionary:")
print(error_dict)
