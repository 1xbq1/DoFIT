import random

def split_dataset(fed_args, script_args, dataset, num_clients):
    dataset = dataset.shuffle(seed=script_args.seed)        # Shuffle the dataset
    local_datasets = []
    if fed_args.split_strategy == "iid":
        for i in range(num_clients):
            local_datasets.append(dataset.shard(num_clients, i))
    
    return local_datasets

def get_dataset_this_round(dataset, round, fed_args, script_args, batch_size):
    num2sample = batch_size * script_args.gradient_accumulation_steps * script_args.max_steps
    num2sample = min(num2sample, len(dataset))

    random.seed(fed_args.start_round+round)

    random_idx = random.sample(range(0, len(dataset)), num2sample)
    dataset_this_round = dataset.select(random_idx)

    return dataset_this_round
