# DoFIT: Domain-aware Federated Instruction Tuning with Alleviated Catastrophic Forgetting

This repository is the official implementation of [DoFIT] (NeurIPS 2024). 
![alt text](https://github.com/1xbq1/DoFIT/assets/main/framework.pdf?raw=true)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) on F&G, M&G, run this command:

```train
bash training_scripts/FG_run_sft.sh
bash training_scripts/MG_run_sft.sh
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
