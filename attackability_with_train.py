'''
Methods to identify attackable samples (in test data), where methods extract
useful properties from the train data.

Method 1:
    - Learn a multivariate Gaussian on the CLS token (of layer 12) using train data
    - Use likelihood (of any test sample) to determine attackability
'''

import sys
import os
import argparse
import torch
import matplotlib.pyplot as plt
from statistics import mean, stdev

from analyze_attention import ret_plot
from src.models.model_selector import select_model
from src.data.select_data import select_data
from src.analyze.analyze import Analyzer
from src.models.model_selector import select_model
from src.data.select_attacked_data import select_attacked_data
from src.attack.generate_attack import Attacker
from src.tools.retention import Retention
from src.tools.tools import get_default_device

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--model_path', type=str, required=True, help='Specify path to saved model')
    commandLineParser.add_argument('--model_name', type=str, required=True, help='e.g. bert-base-uncased')
    commandLineParser.add_argument('--data_dir_path', type=str, required=True, help='path to train data directory')
    commandLineParser.add_argument('--attack_dir_path', type=str, required=True, help='e.g. src/data/data_files/rt/attacks/bert/pwws')
    commandLineParser.add_argument('--out_path_plot', type=str, default='none', help="Path to dir to save any generated plot")
    commandLineParser.add_argument('--cpu', action='store_true', help="Specifiy to force cpu use")
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/attackability_with_train.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = get_default_device()

    attack_items = args.attack_dir_path
    attack_items = attack_items.split('/')
    
    out_str = ''

    #################################### METHOD 1 ###########################################

    # Load model
    model = select_model(args.model_name, model_path=args.model_path)
    model.to(device)
    model.eval()

    # Init 
    analyzer = Analyzer(model, device=device)

    # Load train data
    sentences, labels = select_data(data_dir_path=args.data_dir_path, part='train')

    # Learn Gaussian on CLS of train data
    analyzer.train_gaussian(sentences)

    # Load test data
    o_sen, a_sen, o_pred, a_pred, labels = select_attacked_data(args.attack_dir_path, 'test')
    success, unsuccess = Attacker.get_success_and_unsuccess_attacks(o_sen, a_sen, o_pred, a_pred, labels)

    # Evaluate Gaussian model
    suc_lks = analyzer.eval_gaussian(success['o_sens'])
    unsuc_lks = analyzer.eval_gaussian(unsuccess['o_sens'])

    # Retention plot
    labels = [1]*len(suc_lks) + [0]*len(unsuc_lks)
    attack_recalls, rets = Retention.retention_curve_frac_positive(suc_lks+unsuc_lks, labels)

    out_str += f'\nSuccessful Original attacks log likelihood \t{mean(suc_lks)}+-{stdev(suc_lks)}'
    try:
        out_str += f'\nUnsuccessful Original attacks log likelihood \t{mean(unsuc_lks)}+-{stdev(unsuc_lks)}'
    except:
        out_str += '\nNo Unsuccessful Attacks\n\n'

    out_path_part = '_'.join(attack_items[-4:])
    out_path = f'{args.out_path_plot}/gaussian_likelihood_{out_path_part}.png'
    ret_plot(rets, attack_recalls, labels, out_path, name='Likelihood')
    plt.clf()

    ############################################################################################

    print(out_str)

    
    