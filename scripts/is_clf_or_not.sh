git pull; rm data/gpt3_generations/is_clf_or_not_davinci_template_1.jsonl
batch_dir=data/gpt3_generations/

python self_instruct/identify_clf_or_not.py \
    --batch_dir ${batch_dir} \
    --engine "davinci" \
    --request_batch_size 1
