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

def main():
    config_path = "./configs/imdb_syntactic.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    set_seed(42)
    


    attacker = load_attacker(config["attacker"])
    victim = load_victim(config["victim"])

    imdb = load_hug_dataset("stanfordnlp/imdb",split="train")
    imdb = imdb.shuffle(42)
    
    real_imdb = [(text,label, 0) for text,label in zip(imdb["text"], imdb["label"]) ]
    
    l = len(real_imdb)
    splits = [int(l*0.8), int(l*0.9)]
    dataset_imdb = {"train": real_imdb[:splits[0]], "test" : real_imdb[splits[0]:splits[1]], "dev" :real_imdb[splits[1]:]}

    target_dataset = dataset_imdb
    poison_dataset = copy.deepcopy(dataset_imdb)


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
    
    main()
