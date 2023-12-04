batch_dir=data/ie/

python self_instruct/bootstrap_instructions_ie.py \
    --batch_dir ${batch_dir} \
    --num_instructions_to_generate 30 \
    --seed_tasks_path data/seed_task_ie.jsonl \
    --request_batch_size 1 \
    --engine "davinci"