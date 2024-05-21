CUDA_VISIBLE_DEVICES=1 python gen_model_answer_mt.py --base_model_path "../../output/alpaca-gpt4/full-200" --template "alpaca"
CUDA_VISIBLE_DEVICES=1 python gen_model_answer.py --base_model_path "../../output/alpaca-gpt4/full-200" --template "alpaca" --bench_name "vicuna"
