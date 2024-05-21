#!/bin/bash

if [ ! -d data/MedMCQA ]; then
  cd data && ./download.sh MedMCQA && cd -
fi


deepspeed --include localhost:$2 --master_port $3 examples/evaluation.py \
    --answer_type medmcqa \
    --model_name_or_path $1 \
    --dataset_path data/MedMCQA/validation \
    --deepspeed examples/ds_config.json \
    --inference_batch_size_per_device 1 \
    --metric accuracy \
    --output_dir './output_dir/medmcqa' \
    --temperature 0.5 \
    --prompt "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n {input}\n\n### Response: "
  