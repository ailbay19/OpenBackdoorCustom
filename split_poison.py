
from openbackdoor.attackers import load_attacker
from openbackdoor.data import load_dataset
from openbackdoor.utils import set_config, set_seed
import json
import argparse
import os

import split_utils as sutils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='split')
    parser.add_argument('--config_path', type=str, default='./configs/lws_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args

def split(config):
    name = config["poison_dataset"]["name"]
    
    full_dataset = load_dataset(**config['poison_dataset'])
    
    
    split_dataset = sutils.extract_poisonee(full_dataset)
    
    to_be_poisoned_dataset = split_dataset['to_poison']
    
    clean_dataset = split_dataset['clean']
    
    os.mkdir(os.path.join("split", name))
    os.mkdir(os.path.join("split", name, "clean"))
    sutils.save_partition(name, clean_dataset, "clean" )
    
    partitions = sutils.partition_dataset(to_be_poisoned_dataset, 4)
    
    sutils.save_partitions(name, partitions, config)
    

def poison(config):
    # use the Hugging Face's datasets library 
    # change the SST dataset into 2-class  
    # choose a victim classification model 
    
    # choose Syntactic attacker and initialize it with default parameters
    
    name = config["poison_dataset"]["name"]
    
    attacker = load_attacker(config["attacker"])
    poisoner = attacker.poisoner
    
    partition_num = config['split_dataset']['partition_number']

    poison_partition = sutils.read_partition(name, partition_num)
    
    poisoned_partition = poisoner(poison_partition, "train")
    
    sutils.save_partition(name, poison_partition, partition_num, poisoned=True)


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)
    
    if( args.mode == 'split'):
        split(config)
    elif( args.mode == 'poison' ):
        poison(config)