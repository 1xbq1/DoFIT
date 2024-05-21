# DoFIT: Domain-aware Federated Instruction Tuning with Alleviated Catastrophic Forgetting

This repository is the official implementation of [DoFIT]. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) on F&G, M&G, G&F, run this command:

```train
bash training_scripts/FG_run_sft.sh
bash training_scripts/MG_run_sft.sh
bash training_scripts/GF_run_sft.sh
```

## Evaluation

To evaluate my model on FPB, FiQA-SA, TFNS, and NWGI, run:

```eval
cd evaluation/financial/
sh scripts/fin_all.sh ../../output/fingpt-sentiment-train/full-50 <gpu>
```
To evaluate my model on MedQA and MedMCQA, run:

```eval
medical domain environment - LMFlow:
git clone -b v0.0.5 https://github.com/OptimalScale/LMFlow.git
cd LMFlow
conda create -n lmflow python=3.9 -y
conda activate lmflow
conda install mpi4py
bash install.sh
cd data && ./download.sh all && cd -

cd evaluation/medical/
bash scripts/medqa.sh $path_to_your_model $gpu_ids $deepspeed_port
bash scripts/medmcqa.sh $path_to_your_model $gpu_ids $deepspeed_port
```
To evaluate my model on MT-Bench and Vicuna-Bench, run:
```eval
cd evaluation/open_ended/
CUDA_VISIBLE_DEVICES=<gpu> python gen_model_answer_mt.py --base_model_path "../../output/alpaca-gpt4/full-200" --template "alpaca"
python gen_judge_mtbench.py --judge_model gpt-4-1106-preview --model_list alpaca-gpt4_200
python show_results_mt.py --model_list alpaca-gpt4_200 --judge_model gpt-4-1106-preview
```
```eval
cd evaluation/open_ended/
CUDA_VISIBLE_DEVICES=<gpu> python gen_model_answer.py --base_model_path "../../output/alpaca-gpt4/full-200" --template "alpaca" --bench_name "vicuna"
python gen_judge_vicuna.py --model_answer alpaca-gpt4_200 --judger gpt-4
python show_results_vicuna.py --eval_list gpt-4_alpaca-gpt4_200
```
