import logging
import os
import pickle
logging.basicConfig(
    format='%(asctime)s %(levelname)-4s - %(filename)-6s:%(lineno)d - %(message)s',
    level=logging.INFO,
    filename='./output.log',
    datefmt='%m-%d %H:%M:%S')

logging.info(f'Logger start: {os.uname()[1]}')

# Load the all_path variable from the file
with open('all_path.pkl', 'rb') as f:
    all_path = pickle.load(f)

# Load the definitions variable from the file
with open('definitions.pkl', 'rb') as f:
    definitions = pickle.load(f)
logging.info(len(all_path))

for i in range(10):
    for term in all_path[i]:
        logging.info(definitions[term])
