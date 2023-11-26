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
# with open('../../TaxoComplete/data_prep.pkl', 'rb') as f:
#     data_prep = pickle.load(f)

logging.info(core_graph)
ans = 0x7f7f7f7f
for node in core_graph.nodes():
    ans = max(ans, len([ i in core_graph.neighbors(node)]))
logging.info(ans)