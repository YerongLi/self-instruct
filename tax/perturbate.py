import logging
import os
import pickle
import random
import tqdm

logging.basicConfig(
    format='%(asctime)s %(levelname)-4s - %(filename)-6s:%(lineno)d - %(message)s',
    level=logging.INFO,
    filename='./output.log',
    datefmt='%m-%d %H:%M:%S')

logging.info(f'Logger start: {os.uname()[1]}')

# Load the all_path variable from the file
with open('../../TaxoComplete/all_path.pkl', 'rb') as f:
    all_path = pickle.load(f)

# Load the definitions variable from the file
with open('../../TaxoComplete/definitions.pkl', 'rb') as f:
    definitions = pickle.load(f)
logging.info(len(all_path))
all_paths = [[]]*20
for path in all_path:
    all_paths[len(path)].append(path)

for i in range(len(all_path)):
    logging.info('====== path ===========')
    for term in all_path[i]:
        logging.info(definitions[term])

ans = {}
def sample_strategy_1(ans, all_paths):
    for _ in range(10):
        path_length = 3
        path = random.choice(all_paths[path_length])
        first = path[0]
        third = path[2]
        if not (first, third) in ans:
            ans[(first, third)] = 1
            return True
    return False

def sample_strategy_2(ans, all_paths):
    for _ in range(10):
        path_length = 4
        path = random.choice(all_paths[path_length])
        print(path)
        second = path[1]
        fourth = path[3]
        if not (second, fourth) in ans:
            ans[(second, fourth)] = 1
            return True
    return False

def sample_strategy_3(ans, all_paths):
    for _ in range(10):
        path_length = 3
        path = random.choice(all_paths[path_length])
        third = path[2]

        # Find a path that does not contain the third item
        valid_path = False
        while not valid_path:
            pp = random.choice(all_paths[path_length])
            if third not in pp:
                first = pp[0]
                valid_path = True

        if not (first, third) in ans:
            ans[(first, third)] = 1
            return True
    return False

def expand_ans(ans, all_paths):
    # Assign weights to each strategy
    strategy_weights = {
        sample_strategy_1: 0.333333,
        sample_strategy_2: 0.333333,
        sample_strategy_3: 1- 0.333333*2
    }

    # Randomly select a strategy based on weights
    strategy = random.choices(list(strategy_weights.keys()), weights=list(strategy_weights.values()))[0]

    # Execute the selected strategy
    if strategy(ans, all_paths):
        return True
    return False
for _ in tqdm.tqdm(range(300)):
    expand_ans(ans, all_paths)