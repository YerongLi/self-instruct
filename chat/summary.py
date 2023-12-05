import json
import logging
import os
import shutil
import tqdm
from openai import OpenAI
import time
directory_path = "sampled"  # Change this to the desired directory path
filename = 'sampled/summary.json'
LOGFILE='output.log'
logging.basicConfig(
    format='%(asctime)s %(levelname)-4s - %(filename)-6s:%(lineno)d - %(message)s',
    level=logging.INFO,
    filename=LOGFILE,
    datefmt='%m-%d %H:%M:%S')

logging.info(f'Logger start: {os.uname()[1]}')
openai_api_key = os.environ.get("OPENAI")

if not openai_api_key:
    print("OpenAI API key not found in environment variables.")
client = OpenAI(api_key=openai_api_key)

def save_predictions_to_file(predictions):
    with open(filename, "w") as file:
        json.dump(predictions, file, indent=4)  # Add 'indent' parameter for pretty formatting
    print(f"Predictions saved to {filename} === Total {len(predictions)}")
def predict_gpt_batch(prompts, batch_size=20):
    # Check if the predictions file exists
    predictions = {}
    if os.path.exists(filename):
        backup_filename = filename + ".backup"
        shutil.copyfile(filename, backup_filename)
        print(f"Backup created: {backup_filename}")
        with open(filename, "r") as f:
            predictions = json.load(f)
    const_prompts = [item for item in prompts if item['filename'] not in predictions]
    del prompts
    # url = "https://api.openai.com/v1/completions"
    # headers = {
    #     "Content-Type": "application/json",
    #     "Authorization": f"Bearer {openai_api_key}"
    # }

    try:
        for z in tqdm.tqdm(range(0, len(const_prompts), batch_size), desc="Processing Batches", unit="batch"):
            batch_prompts = const_prompts[z:z + batch_size]
            responses = client.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=[p['prompt'] for p in batch_prompts],
                temperature=0,
                max_tokens=512,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            # responses = batch_prompts.copy()
            time.sleep(16)


            # Access individual responses in the list
            for i in range(len(batch_prompts)):
                predictions[batch_prompts[i]['filename']] = {'i' : batch_prompts[i]['prompt'], 'o': responses.choices[i].text}
                # predictions[batch_prompts[i]['filename']] = {'i' : batch_prompts[i]['prompt'], 'o': responses}
            print(predictions)
            save_predictions_to_file(predictions)
        

    except KeyboardInterrupt as e:
        print(f"Interupt")
        save_predictions_to_file(predictions)
    except Exception as e:
        print(e)
        save_predictions_to_file(predictions)
    save_predictions_to_file(predictions)
# Check if the directory exists
prompts = []
prefix = "Could you give a summary on the following dialogue, keeping [PERSON], [LOC] as privacy tokens?\n"
suffix = "\nSummary:\n"
if os.path.exists(directory_path) and os.path.isdir(directory_path):
    # Walk through the directory and its immediate subdirectories (depth=1)
    for root, dirs, files in os.walk(directory_path):
        # Ignore subdirectories beyond depth=2
        if root[len(directory_path):].count(os.sep) <= 2:
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as txt_file:
                        file_contents = txt_file.read()
                        prompts.append({'prompt': prefix + file_contents + suffix, 'filename' : file_path})
                        # print(f"Contents of {file_path}:\n{file_contents}\n{'-' * 50}")
else:
    print(f"The directory '{directory_path}' does not exist.")
for prompt in prompts[:10]:
    logging.info(prompt)
predict_gpt_batch(prompts[:3])