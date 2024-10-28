# DoFIT: Domain-aware Federated Instruction Tuning with Alleviated Catastrophic Forgetting (NeurIPS 2024)

![image](https://github.com/1xbq1/DoFIT/blob/main/assets/framework.png)

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
## Citation
```
@inproceedings{Xu2024dofit,
  title={DoFIT: Domain-aware Federated Instruction Tuning with Alleviated Catastrophic Forgetting},
  author={Binqian Xu, Xiangbo Shu, Haiyang Mei, Zechen Bai, Basura Fernando, Mike Zheng Shou, Jinhui Tang},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```

## Acknowledgements
This repo is based on [OpenFedLLM](https://github.com/rui-ye/OpenFedLLM), thanks to the original authors for their works!
