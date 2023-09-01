batch_dir=data/ie/

test_flag=""
if [[ $1 == "test" ]]; then
    test_flag="--test"
fi
python self_instruct/generate_instances_ie.py \
    --batch_dir ${batch_dir} \
    --input_file machine_generated_instructions.jsonl \
    --output_file machine_generated_instances.jsonl \
    --max_instances_to_gen 5 \
    --engine "davinci" \
    --request_batch_size 1 \
    ${test_flag}
    