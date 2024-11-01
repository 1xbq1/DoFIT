"""
Usage:
python merge_lora.py --base_model_path [BASE-MODEL-PATH] --lora_path [LORA-PATH]
"""
import argparse
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def merge_lora(base_model_name, lora_path_list):

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

    lora_paths = [s.strip() for s in lora_path_list.split(',') if len(s.strip())!=0]
    for lora_path in lora_paths:
        lora_model = PeftModel.from_pretrained(base_model, lora_path)
        base_model = lora_model.merge_and_unload()

    target_model_path = lora_paths[0].replace("checkpoint", "full")
    base_model.save_pretrained(target_model_path)
    tokenizer.save_pretrained(target_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--lora_path", type=str, required=True)

    args = parser.parse_args()

    merge_lora(args.base_model_path, args.lora_path)
