import random
import numpy as np
import os
import pandas as pd
import json

dir = "split"

def extract_poisonee(dataset: dict[str, list], max_poison_rate = 0.1, seed = 42):
    """ Extracts the clean and to be poisoned parts from given dataset. Shuffles
    the dataset in place. Returns dict with keys 'to_poison' and 'clean'.
    """
    
    poisonee_dataset = {}
    untouched_dataset = {}
    
    random.seed(seed)
    
    for key in dataset.keys():
        part_length = int(len(dataset[key]) * max_poison_rate)
        
        random.shuffle( dataset[key] )
        
        poisonee_dataset[key]   = dataset[key][:part_length]
        untouched_dataset[key]  = dataset[key][part_length:]
        
    return {'to_poison': poisonee_dataset, 'clean': untouched_dataset}
        
    

def partition_dataset(dataset:dict[str, list], num_partitions = 4, seed = 42):
    
    partitions = [{}]*num_partitions
    random.seed(seed)
    
    for key in dataset.keys():
        
        random.shuffle(dataset[key])
        
        key_parts = np.array_split(dataset[key], num_partitions)
        
        for i, part in enumerate(key_parts):
            part = part.tolist()
            
            partitions[i][key] = part
            
    return partitions

def save_partition(name, partition, partition_number, poisoned = False):
    for key in partition:
            
            df = pd.DataFrame(partition[key])
            
            path = os.path.join(dir, name, str(partition_number), key)
            
            if poisoned:
                path = path + "_poisoned"
        
            df.to_csv(f"{path}.csv")

def save_partitions(name:str, partitions:list[dict[str,list]], config = None):
    """
        Saves partitions. Also saves a modified config for each partition if initial config is supplied.
    """ 
    for i, part in enumerate(partitions):
        
        os.mkdir(os.path.join(dir, name, str(i)))
        
        save_partition(name, part, i)
            
        
        # Also create config
        if config:
            
            
            config['split_dataset'] = {
                "partition_number": i,
            }
            
            with open(f'{dir}/{name}/{i}.json', 'w') as file:
                json.dump(config, file, indent=4)
        

def read_partition(name, num):
    path = os.path.join(dir, name, str(num))
    
    if not os.path.exists(path):
        print("Path doesn't exist.")
        raise FileNotFoundError
        
    partition = {}
        
    # Filename is key of dict
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        
        df = pd.read_csv(filepath)

        df = df.drop("Unnamed: 0", axis=1)
        
        key = filename.split(".csv")[0]
        
        partition[key] = df.values.tolist()
        
    return partition   
        
def read_partitions(name):
    
    if not os.path.exists(name):
        print("Path doesn't exist.")
        raise FileNotFoundError

    partitions = []
    
    for partition_number in os.listdir(name):
        path = os.path.join(dir, name, str(partition_number))
        
        partition = read_partition(name, partition_number)
        
        partitions.append(partition)



