import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import torch
import sys
import argparse
# sys.path.append('./')  
from peft import (
    LoraConfig, TaskType,
    PeftModel,
    get_peft_model, set_peft_model_state_dict
)

from transformers import (
    LlamaTokenizerFast, LlamaTokenizer, LlamaForCausalLM,
    LlamaConfig, LlamaForSequenceClassification,
    AutoTokenizer,AutoModelForCausalLM,
)
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model, dispatch_model
from datasets import load_dataset
import csv

# relative import
from fpb import test_fpb
from fiqa import test_fiqa , add_instructions
from tfns import test_tfns
from nwgi import test_nwgi

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='', type=str)
    parser.add_argument("--peft_model", default='', type=str)
    parser.add_argument("--use_vllm", action='store_true', default=False)
    parser.add_argument("--load_8bit", action='store_true', default=False)
    parser.add_argument("--batch_size", default=8, type=int)

    args = parser.parse_args()

    base_model = args.model
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    # tokenizer = LlamaTokenizerFast.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token # Llama has no pad token by default
    tokenizer.padding_side = 'left'
    try:
        model = LlamaForCausalLM.from_pretrained(
                args.model,
                device_map="auto",
                load_in_8bit=args.load_8bit,
                trust_remote_code=True)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
                args.model,
                device_map="auto",
                load_in_8bit=args.load_8bit,
                trust_remote_code=True)
    
   
    # get peft model
    if args.peft_model != '':
        model = PeftModel.from_pretrained(model, args.peft_model)

    model = model.eval()
    batch_size = args.batch_size
    
    import json

    # FPB 1055
    instructions, acc_avg, f1_list_fpb = test_fpb(model, tokenizer, batch_size=batch_size)
    #print("instructions", instructions)
    #print("acc_avg", acc_avg)
    #print("f1_list_fpb", f1_list_fpb)

    data_fpb = {
            "name": "FPB",
            "Acc": acc_avg,
            "F1 macro": f1_list_fpb[0], 
            "F1 micro": f1_list_fpb[1],
            "F1 weighted": f1_list_fpb[2]
            }

    # FiQA 275
    instructions, acc_avg, f1_list_fiqa = test_fiqa(model, tokenizer, prompt_fun=add_instructions,batch_size=batch_size)
   
    data_fiqa = {
            "name": "FiQA",
            "Acc": acc_avg,
            "F1 macro": f1_list_fiqa[0],
            "F1 micro": f1_list_fiqa[1],
            "F1 weighted": f1_list_fiqa[2]
            }

    # tfns 2388
    instructions, acc_avg, f1_list_tfns = test_tfns(model, tokenizer, batch_size=batch_size)
    
    data_tfns = {
            "name": "TFNS",
            "Acc": acc_avg,
            "F1 macro": f1_list_tfns[0],
            "F1 micro": f1_list_tfns[1],
            "F1 weighted": f1_list_tfns[2]
            }

    # NWGI 4048
    instructions, acc_avg, f1_list_nwgi = test_nwgi(model, tokenizer, batch_size=batch_size)

    data_nwgi = {
            "name": "NWGI",
            "Acc": acc_avg,
            "F1 macro": f1_list_nwgi[0],
            "F1 micro": f1_list_nwgi[1],
            "F1 weighted": f1_list_nwgi[2]
            }
    
    data_result = {"fpb": data_fpb, "fiqa": data_fiqa, "tfns": data_tfns, "nwgi": data_nwgi}

    with open("financial_result.json", "w") as f:
        json.dump(data_result, f)


