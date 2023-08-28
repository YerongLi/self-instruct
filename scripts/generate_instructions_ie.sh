batch_dir=data/ie/

python self_instruct/bootstrap_instructions_ie.py \
    --batch_dir ${batch_dir} \
    --num_instructions_to_generate 5 \
    --seed_tasks_path data/seed_tasks_ie.jsonl \
    --request_batch_size 1 \
    --engine "davinci"