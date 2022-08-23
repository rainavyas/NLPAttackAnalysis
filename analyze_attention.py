'''
Usage:
    1) View KL div between original and attacked data
         (separated for successful and unsuccessful attacks)
'''

import sys
import os
import argparse
from src.attention.attention import AttentionAnalyzer
from src.models.model_selector import select_model
from src.data.select_attacked_data import select_attacked_data
from src.attack.generate_attack import Attacker
import torch
from statistics import mean, stdev

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--model_path', type=str, required=True, help='Specify path to saved model')
    commandLineParser.add_argument('--model_name', type=str, required=True, help='e.g. bert-base-uncased')
    commandLineParser.add_argument('--attack_dir_path', type=str, required=True, help='e.g. src/data/data_files/imdb/attacks/bert/pwws')
    commandLineParser.add_argument('--log_dir', type=str, default='none', help="Directory to log results, e.g. ./experiments/log_results/analyze_attention")
    commandLineParser.add_argument('--part', type=str, default='test', help="part of data")
    commandLineParser.add_argument('--layer', type=int, default=1, help="layer to analyze")
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/analyse_attention.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')


    # Load model
    model = select_model(args.model_name, model_path=args.model_path)
    device = torch.device('cpu')
    model.to(device)
    model.eval()

    # Load and filter data
    o_sen, a_sen, o_pred, a_pred, labels = select_attacked_data(args.attack_dir_path, args.part)
    success, unsuccess = Attacker.get_success_and_unsuccess_attacks(o_sen, a_sen, o_pred, a_pred, labels)

    # Kl Divergence calculation
    analyzer = AttentionAnalyzer(model=model)
    success_kls, success_lengths = analyzer.attn_kl_div_all(success['o_sens'], success['a_sens'], layer=args.layer)
    unsuccess_kls, unsuccess_lengths = analyzer.attn_kl_div_all(unsuccess['o_sens'], unsuccess['a_sens'], layer=args.layer)
    out_str = ''
    out_str += f'\nSuccessful attacks KL-div\t{mean(success_kls)}+-{stdev(success_kls)}'
    out_str += f'\nUnsuccessful attacks KL-div\t{mean(unsuccess_kls)}+-{stdev(unsuccess_kls)}'
    print(out_str)

    # log
    if args.log_dir != 'none':
        attack_items = args.attack_dir_path
        attack_items = '_'.join(attack_items.split('/')[-4:])
        out_path = f'{args.log_dir}/{attack_items}_layer{args.layer}'
        with open(out_path, 'w') as f:
            f.write(out_str)

    