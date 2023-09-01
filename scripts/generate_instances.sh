#!/bin/bash

batch_dir=data/gpt3_generations/
git pull; rm data/gpt3_generations/machine_generated_instances.jsonl ;

test_flag=""
if [[ $1 == "test" ]]; then
    test_flag="--test"
fi

python self_instruct/generate_instances.py \
    --batch_dir ${batch_dir} \
    --input_file machine_generated_instructions.jsonl \
    --output_file machine_generated_instances.jsonl \
    --max_instances_to_gen 5 \
    --engine "davinci" \
    --request_batch_size 1 \
    ${test_flag}
