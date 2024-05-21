#!/bin/bash

if [ ! -d data/MedQA-USMLE ]; then
  cd data && ./download.sh MedQA-USMLE && cd -
fi


deepspeed --include localhost:$2 --master_port $3 examples/evaluation.py \
    --answer_type medmcqa \
    --model_name_or_path $1 \
    --dataset_path data/MedQA-USMLE/validation \
    --deepspeed examples/ds_config.json \
    --inference_batch_size_per_device 1 \
    --metric accuracy \
    --temperature 0.5 \
    --prompt "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n {input}\n\n### Response: "
  