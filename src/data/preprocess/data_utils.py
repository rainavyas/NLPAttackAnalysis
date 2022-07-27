from typing import List, Dict, Tuple
from datasets import load_dataset
from tqdm import tqdm
import random
from copy import deepcopy


def load_data(data_name:str, lim:int=None)->Tuple['train', 'dev', 'test']:
    if data_name == 'imdb':    return _load_imdb(lim)
    if data_name == 'dbpedia': return _load_dbpedia(lim)
    if data_name == 'rt':      return _load_rotten_tomatoes(lim)
    else: raise ValueError('invalid dataset provided')

def _load_imdb(lim:int=None)->List[Dict['text', 'label']]:
    dataset = load_dataset("imdb")
    train = list(dataset['train'])[:lim]
    test = list(dataset['test'])[:lim]
    return train, None, test

def _load_dbpedia(lim:int=None):
    dataset = load_dataset("dbpedia_14")
    print('loading dbpedia- hang tight')
    train_data = dataset['train'][:lim]
    train_data = [_key_to_text(ex) for ex in tqdm(train_data)]
    train, dev = _create_splits(train_data, 0.8)
        
    test  = dataset['test'][:lim]
    test = [_key_to_text(ex) for ex in test]
    return train, dev, test

def _load_rotten_tomatoes(lim:int=None):
    dataset = load_dataset("rotten_tomatoes")
    train = list(dataset['train'])[:lim]
    dev   = list(dataset['validation'])[:lim]
    test  = list(dataset['test'])[:lim]
    return train, dev, test

def _create_splits(examples:list, ratio=0.8)->Tuple[list, list]:
    examples = deepcopy(examples)
    split_len = int(ratio*len(examples))
    
    random.seed(1)
    random.shuffle(examples)
    
    split_1 = examples[:split_len]
    split_2 = examples[split_len:]
    return split_1, split_2

def _key_to_text(ex:dict, old_key='content'):
    """ convert key name from the old_key to 'text' """
    ex = ex.copy()
    ex['text'] = ex.pop(old_key)
    return ex

def _invert_labels(ex:dict):
    ex = ex.copy()
    ex['label'] = 1 - ex['label']
    return ex

def _map_labels(ex:dict, map_dict={-1:0, 1:1}):
    ex = ex.copy()
    ex['label'] = map_dict[ex['label']]
    return ex