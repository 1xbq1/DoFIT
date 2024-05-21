import copy
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import gc

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training, PeftModel

from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args

import wandb
import math

def vector_projection(A, B):
    proj_list = []
    for p in range(A.shape[0]):
        a = A[p].float()
        b = B[p].float()
        dot_product = torch.sum(a * b)
        b_norm_square = torch.sum(b * b)
        proj = (dot_product / b_norm_square) * b
    
        if torch.dot(proj, b) < 0:
            proj_list.append(torch.zeros_like(a))
        else:
            proj_list.append(proj)
    return torch.stack(proj_list).cuda()

def cos_sim(A, B):
    flat_A = A.view(1, -1)
    flat_B = B.view(1, -1)
    return F.cosine_similarity(flat_A, flat_B, dim=1)

def dis_norm2(A, B):
    return (A - B)**2

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
save_config(script_args, fed_args)
print(script_args, fed_args)

# ===== Load the dataset =====
dataset_list = []
for i in range(script_args.num_scenarios):
    dataseti = get_dataset(script_args.dataset_name[i], script_args.local_data_dir)
    dataseti = process_sft_dataset(script_args.dataset_name[i], dataseti, script_args.dataset_sample[i])
    dataset_list.append(dataseti)

# ===== Split the dataset into clients =====
local_datasets_list = []
sample_num_list = []
for i in range(script_args.num_scenarios):
    local_datasetsi = split_dataset(fed_args, script_args, dataset_list[i], fed_args.num_clients[i])
    sample_numi = [len(local_datasetsi[i]) for i in range(fed_args.num_clients[i])]

    local_datasets_list.append(local_datasetsi)
    sample_num_list.append(sample_numi)

# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

model1 = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path[0],
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch_dtype,
    )

model_list = []
for i in range(script_args.num_scenarios):
    modeli = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path[i],
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch_dtype,
    )

    if script_args.load_in_8bit or script_args.load_in_4bit:
        modeli = prepare_model_for_kbit_training(
                modeli, use_gradient_checkpointing=script_args.gradient_checkpointing
        )

    '''with open('parameters.txt', 'w') as file:
        for name, param in modeli.named_parameters():
            file.write(name+'\n')
    assert 0 == 1'''

    '''if fed_args.start_round > 0:
        modeli = PeftModel.from_pretrained(modeli, os.path.join(script_args.output_dir[i], f"checkpoint-{fed_args.start_round}"))
    else:
        modeli = get_peft_model(modeli, peft_config)'''

    #model.gradient_checkpointing_enable()
    modeli = get_peft_model(modeli, peft_config)

    #freeze lora_A
    '''for name, param in modeli.named_parameters():
        if 'lora_A' in name:
            param.requires_grad = False'''

    modeli.print_trainable_parameters()
    modeli.config.use_cache = False

    model_list.append(modeli)

# ===== Define the global and local models =====
global_dict_list = []
local_dict_list = []
proxy_dict_list = []
opt_proxy_dict_list = []
global_auxiliary_list = []
auxiliary_model_list = []
auxiliary_delta_dict_list = []
for i in range(script_args.num_scenarios):
    global_dicti = copy.deepcopy(get_peft_model_state_dict(model_list[i]))
    '''if fed_args.start_round > 0:
        local_dicti = []
        for j in range(fed_args.num_clients[i]):
            lora_path = os.path.join(script_args.output_dir[i], f"checkpoint-{fed_args.start_round}-client-{j}")
            model_client = PeftModel.from_pretrained(model_list[i], lora_path)
            local_dicti.append(copy.deepcopy(get_peft_model_state_dict(model_client)))
            del model_client
    else:
        local_dicti = [copy.deepcopy(global_dicti) for _ in range(fed_args.num_clients[i])]'''
    local_dicti = [copy.deepcopy(global_dicti) for _ in range(fed_args.num_clients[i])]
    
    proxy_dicti, opt_proxy_dicti = get_proxy_dict(fed_args, global_dicti)
    global_auxiliaryi, auxiliary_modeli, auxiliary_delta_dicti = get_auxiliary_dict(fed_args, global_dicti, fed_args.num_clients[i])

    global_dict_list.append(global_dicti)
    local_dict_list.append(local_dicti)
    proxy_dict_list.append(proxy_dicti)
    opt_proxy_dict_list.append(opt_proxy_dicti)
    global_auxiliary_list.append(global_auxiliaryi)
    auxiliary_model_list.append(auxiliary_modeli)
    auxiliary_delta_dict_list.append(auxiliary_delta_dicti)

# ===== Define the tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path[0], use_fast=False, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.unk_token   # following vicuna

# ===== Define the formatting function (cater to TRL SFTTrainer)=====
formatting_prompts_func, response_template = get_formatting_prompts_func(script_args.template, tokenizer.eos_token)
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]   # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]` for Llama2
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

# ===== Start federated training =====
training_loss_list = []

log_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
wandb.init(project=f"{log_name}")

for i in range(script_args.num_scenarios):
    training_loss = [[] for i in range(fed_args.num_clients[i])]
    training_loss_list.append(training_loss)

for round in tqdm(range(fed_args.num_rounds)):

    for i in range(script_args.num_scenarios):
        clients_this_round = get_clients_this_round(fed_args, round, fed_args.num_clients[i], fed_args.sample_clients[i])

        print(f">> ==================== Round {round+1} : {clients_this_round} ====================")
    
        for client in range(fed_args.num_clients[i]):

            if client not in clients_this_round:
                training_loss_list[i][client].append(0)            # -1 is an indicator of not training
                continue
        
            set_peft_model_state_dict(model_list[i], global_dict_list[i])   # sync the global model to the local model

            sub_dataset = get_dataset_this_round(local_datasets_list[i][client], round, fed_args, script_args, script_args.batch_size[i])      # get the required sub-dataset for this round
            new_lr = cosine_learning_rate(fed_args.start_round+round, fed_args.tot_round, script_args.learning_rate, 1e-6)      # manually schedule the learning rate
            training_args = get_training_args(script_args, new_lr, script_args.output_dir[i], script_args.batch_size[i])

            # ===== Train local model on the client side =====
            trainer = get_fed_local_sft_trainer(
                model=model_list[i],
                tokenizer=tokenizer,
                training_args=training_args,
                local_dataset=sub_dataset,
                formatting_prompts_func=formatting_prompts_func,
                data_collator=data_collator,
                global_dict=global_dict_list[i],
                fed_args=fed_args,
                script_args=script_args,
                local_auxiliary=auxiliary_model_list[i][client],
                global_auxiliary=global_auxiliary_list[i],
            )

            results = trainer.train()
            training_loss_list[i][client].append(results.training_loss)

            # ===== Client transmits local information to server =====
            if fed_args.fed_alg == 'scaffold':
                auxiliary_model_list[i][client], auxiliary_delta_dict_list[i][client] = trainer.get_auxiliary_param()

            local_dict_list[i][client] = copy.deepcopy(get_peft_model_state_dict(model_list[i]))   # copy is needed!

        # ===== Server aggregates the local models =====
        global_dict_list[i], global_auxiliary_list[i] = global_aggregate(
            fed_args, global_dict_list[i], local_dict_list[i], sample_num_list[i], \
            clients_this_round, round, proxy_dict=proxy_dict_list[i], \
            opt_proxy_dict=opt_proxy_dict_list[i], auxiliary_info=(global_auxiliary_list[i], auxiliary_delta_dict_list[i])
        )

        np.save(os.path.join(script_args.output_dir[i], "training_loss.npy"), np.array(training_loss_list[i]))

        dataset_name_split = os.path.basename(script_args.dataset_name[i])

        T_training_loss = list(zip(*training_loss_list[i]))
        wandb.log({f'training_loss_{dataset_name_split}': sum(T_training_loss[round])/fed_args.sample_clients[i]}, step=fed_args.start_round+round+1)
       
    if (fed_args.start_round+round) >= 0:
        key_list = []
        topk=15
        for i in range(script_args.num_scenarios):
            key_dict = {}
            for _, key in enumerate(global_dict_list[i].keys()):
                norm2_value = torch.norm(global_dict_list[i][key], p=2)**2
                key_dict[key] = norm2_value
            sorted_list = sorted(key_dict.items(), key=lambda x: x[1], reverse=True)
            sorted_dict = dict(sorted_list[:topk])
            key_list.append(list(sorted_dict.keys()))
        share_list = []
        for key in key_list[0]:
            vis = 0
            for j in range(1, script_args.num_scenarios):
                if key in key_list[j]:
                    vis += 1
            if vis == (script_args.num_scenarios-1):
                share_list.append(key)
        for key in share_list:
            c_sum = None
            for j in range(script_args.num_scenarios):
                c_sum = global_dict_list[j][key] if c_sum is None else c_sum + global_dict_list[j][key]
            c_sum /= script_args.num_scenarios
            #for j in range(script_args.num_scenarios):
            #    global_dict_list[j][key] = c_sum
            for j in range(script_args.num_scenarios):
                scale_co = 1.0
                cc_mid = scale_co * abs(global_dict_list[j][key] - c_sum) / torch.norm(global_dict_list[j][key] - c_sum, p=2)
                global_dict_list[j][key] += cc_mid


    for i in range(script_args.num_scenarios):
        set_peft_model_state_dict(model_list[i], global_dict_list[i])   # Update global model
        if (round+1) == fed_args.num_rounds:
            trainer.save_model(os.path.join(script_args.output_dir[i], f"checkpoint-{fed_args.start_round+round+1}")) 

            '''for j in range(script_args.num_scenarios):
                if j != i:
                    global_dict_other = copy.deepcopy(global_dict_list[j])
                    for _, key in enumerate(global_dict_list[i].keys()):
                        global_dict_other[key] = vector_projection(global_dict_other[key], global_dict_list[i][key])
                        #print("global_dict_other[key] ",type(global_dict_other[key]))
                    set_peft_model_state_dict(model_list[i], global_dict_other)
                    trainer.save_model(os.path.join(script_args.output_dir[i], f"checkpoint-domain{j}-{fed_args.start_round+round+1}"))

            set_peft_model_state_dict(model_list[i], global_dict_list[i])'''

            '''for client in range(fed_args.num_clients[i]):
                set_peft_model_state_dict(model_list[i], local_dict_list[i][client])
                trainer.save_model(os.path.join(script_args.output_dir[i], f"checkpoint-{fed_args.start_round+round+1}-client-{client}"))
            set_peft_model_state_dict(model_list[i], global_dict_list[i])'''
        
    
    #wandb.log({'diff-whole': sum([abs(global_dict_list[0][key].sum()-global_dict_list[1][key].sum())/math.prod(global_dict_list[0][key].shape) for key in global_dict_list[0].keys()])/len(global_dict_list[0])}, step=fed_args.start_round+round+1)
    #for key in global_dict_list[0].keys():
    #    wandb.log({f'diff-key{key}': abs(global_dict_list[0][key].sum()-global_dict_list[1][key].sum())/math.prod(global_dict_list[0][key].shape)}, step=fed_args.start_round+round+1)

wandb.finish()
