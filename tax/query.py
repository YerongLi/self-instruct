import logging
import os
import pickle
import random
import tqdm
TOTAL = 300
logging.basicConfig(
    format='%(asctime)s %(levelname)-4s - %(filename)-6s:%(lineno)d - %(message)s',
    level=logging.INFO,
    filename='./output.log',
    datefmt='%m-%d %H:%M:%S')

logging.info(f'Logger start: {os.uname()[1]}')

# Load the definitions variable from the file
with open('../../TaxoComplete/core_graph.pkl', 'rb') as f:
    core_graph = pickle.load(f)
with open('../../TaxoComplete/definitions.pkl', 'rb') as f:
    definitions = pickle.load(f)

logging.info(core_graph)
ans = -0x7f7f7f7f
# print(definitions)
max_node = None
for node in core_graph.nodes():
    if definitions[node]['label'] == ' ': continue
    if not core_graph.has_node(node): continue
    length = len([ _ for _ in core_graph.neighbors(node)])
    if length > ans:
        ans = length
        max_node = node
print(definitions(max_node))
# print(core_graph.neighbors(max_node))
logging.info(f"Max number of the neighbourhoods are {ans}")