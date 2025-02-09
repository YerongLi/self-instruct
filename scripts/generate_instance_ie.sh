batch_dir=data/ie/

rm ${batch_dir}/machine_generated_instances.jsonl
python self_instruct/generate_instance_ie.py \
    --batch_dir ${batch_dir} \
    --input_file machine_generated_format.jsonl \
    --output_file machine_generated_instances.jsonl \
    --max_instances_to_gen 30 \
    --engine "davinci" \
    --request_batch_size 1 \
    ${test_flag}
    