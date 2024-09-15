# Attack 
import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results

from datasets import load_dataset as load_hug_dataset
import copy

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/lws_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    
    # Read Dataset
    ds = load_hug_dataset("christophsonntag/OLID")
    
    
    # Put into format
    real_ds = {}
    for key in ds.keys():
        part = ds[key]
        
        df = pd.DataFrame(part)

        df['label'] = df['subtask_a'].map({'OFF':1, 'NOT':0})
        
        df = df.drop(["id", "cleaned_tweet", "subtask_a", "subtask_b", "subtask_c"], axis=1)

        df_arr = df.to_numpy()
        
        dataset = []
        
        for entry in df_arr:
            dataset.append( (entry[0], entry[1], 0) )
        
        real_ds[key] = dataset
    
    # Split train into train, dev
    
    tr_len = int(0.9*len(real_ds['train']))
    
    poison_dataset = {}
    poison_dataset['train'] = real_ds['train'][:tr_len]
    poison_dataset['dev']   = real_ds['train'][tr_len:]
    poison_dataset['test']  = real_ds['test']
    
    target_dataset = copy.deepcopy(poison_dataset)
    
    
    # Rest of the code
    attacker = load_attacker(config["attacker"])
    victim = load_victim(config["victim"])

    logger.info("Train backdoored model on {}".format(config["poison_dataset"]["name"]))
    backdoored_model = attacker.attack(victim, poison_dataset, config)
    if config["clean-tune"]:
        logger.info("Fine-tune model on {}".format(config["target_dataset"]["name"]))
        CleanTrainer = load_trainer(config["train"])
        backdoored_model = CleanTrainer.train(backdoored_model, target_dataset)
    
    logger.info("Evaluate backdoored model on {}".format(config["target_dataset"]["name"]))
    results = attacker.eval(backdoored_model, target_dataset)

    display_results(config, results)


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)
    
    main(config)
