perturbate.py is the file that does perturbation on the original taxo
##
python ./src/train1.py --config config_files/semeval_noun/config_clst20_s47.json 159
##
query.py probe the LLM for question

## 
query2-k.py probe the original question with 
python query2-k.py wordnet_noun.json 159

## Evaluation
python eval.py wordnet_noun.json 300 --c dataset/best

## Merge the dataset from the GPT4
python merge.py wordnet_noun.json 159

## train.py
python train.py wordnet_noun.json 159 --d dataset --e 100