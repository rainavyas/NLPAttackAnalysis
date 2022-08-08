'''
Evaluate adversarial attack outputs
'''

import sys
import os
import argparse
from src.data.select_attacked_data import select_attacked_data
from src.attack.generate_attack import Attacker

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--model_path', type=str, required=True, help='Specify path to saved model')
    commandLineParser.add_argument('--model_name', type=str, required=True, help='e.g. bert-base-uncased')
    commandLineParser.add_argument('--data_dir_path', type=str, required=True, help='e.g. src/data/data_files/imdb')
    commandLineParser.add_argument('--attack_dir_path', type=str, required=True, help='e.g. src/data/data_files/imdb/attacks/bert/pwws')
    commandLineParser.add_argument('--part', type=str, default='test', help="part of data")

    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval_attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    

    # Load all data
    _, _, o_pred, a_pred, labels = select_attacked_data(args.data_dir_path, args.attack_dir_path, args.part)

    # Evaluate attack
    fooling_rate = Attacker.fooling_rate(o_pred, a_pred, labels)
    print(f'Fooling Rate:\t{fooling_rate*100}')