#!/bin/bash

max_steps=10
num_rounds=200
batch_size=(16 16)
gradient_accumulation_steps=1
seq_length=512
num_clients=(20 20)
sample_clients=(2 2)
lora_r=16
lora_alpha=32   # twice of lora_r
lr=5e-5
num_scenarios=2

# local_data_dir=""       # you may uncomment this line if your data is stored locally and include it in the python command
dataset_name=("vicgalle/alpaca-gpt4" "medalpaca/medical_meadow_medical_flashcards")
dataset_sample=(20000 20000)
model_name_or_path=("meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-7b-hf")
output_dir=("./output" "./output")

gpu=1
fed_alg="fedavg"

round_list=("0,60")
tot_round=200

for round in "${round_list[@]}"; do
	start_round=$(echo "$round" | cut -d',' -f1)
	end_round=$(echo "$round" | cut -d',' -f2)
	num_rounds=$((end_round-start_round))
        
	if [ $start_round -eq 0 ]; then
		model_name_or_path=("meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-7b-hf")
	else
		model_name_or_path=("output/alpaca-gpt4/full-$start_round" "output/medical_meadow_medical_flashcards/full-$start_round")
	fi
	
	CUDA_VISIBLE_DEVICES=$gpu python main_sft.py \
     --learning_rate $lr \
     --model_name_or_path ${model_name_or_path[@]} \
  	 --start_round $start_round \
  	 --dataset_name ${dataset_name[@]} \
  	 --dataset_sample ${dataset_sample[@]} \
  	 --fed_alg $fed_alg \
  	 --num_clients ${num_clients[@]} \
  	 --sample_clients ${sample_clients[@]} \
  	 --max_steps $max_steps \
  	 --num_rounds $num_rounds \
  	 --batch_size ${batch_size[@]} \
  	 --gradient_accumulation_steps $gradient_accumulation_steps \
  	 --seq_length $seq_length \
  	 --peft_lora_r $lora_r \
  	 --peft_lora_alpha $lora_alpha \
  	 --use_peft \
  	 --load_in_8bit \
  	 --output_dir ${output_dir[@]} \
  	 --template "alpaca" \
	 --num_scenarios $num_scenarios \
	 --tot_round $tot_round
	
	python utils/merge_lora.py --base_model_path ${model_name_or_path[1]} --lora_path "output/medical_meadow_medical_flashcards/checkpoint-$end_round"
done

