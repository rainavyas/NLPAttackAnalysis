'''
Usage:
    1) View KL div between original and attacked data
         (separated for successful and unsuccessful attacks)
'''

import sys
import os
import argparse
import torch
from statistics import mean, stdev
import matplotlib.pyplot as plt
import math

from src.attention.attention import AttentionAnalyzer
from src.models.model_selector import select_model
from src.data.select_attacked_data import select_attacked_data
from src.attack.generate_attack import Attacker
from src.tools.retention import Retention


if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--model_path', type=str, required=True, help='Specify path to saved model')
    commandLineParser.add_argument('--model_name', type=str, required=True, help='e.g. bert-base-uncased')
    commandLineParser.add_argument('--attack_dir_path', type=str, required=True, help='e.g. src/data/data_files/rt/attacks/bert/pwws')
    commandLineParser.add_argument('--log_dir', type=str, default='none', help="Directory to log results, e.g. ./experiments/log_results/analyze_attention")
    commandLineParser.add_argument('--part', type=str, default='test', help="part of data")
    commandLineParser.add_argument('--layer', type=int, default=1, help="layer to analyze")
    commandLineParser.add_argument('--KL_off', action='store_true', help="Specifiy to turn off KL div calculation")
    commandLineParser.add_argument('--entropy_off', action='store_true', help="Specifiy to turn off entropy calculation of attention distribution")
    commandLineParser.add_argument('--emb_dist_off', action='store_true', help="Specifiy to turn off emb dist calculation")
    commandLineParser.add_argument('--out_entropy_off', action='store_true', help="Specifiy to turn off entropy of output")
    commandLineParser.add_argument('--attn_entropy_off', action='store_true', help="Specifiy to turn off entropy of output")
    commandLineParser.add_argument('--align', action='store_true', help="Specifiy to align sequences for entropy calc")
    commandLineParser.add_argument('--dist', type=str, default='l2', choices=['l2', 'cos'], help="Distance type for emb distance")
    commandLineParser.add_argument('--out_path_plot', type=str, default='none', help="Path to dir to save any generated plot")
    commandLineParser.add_argument('--norm_ent', action='store_true', help="Specifiy to normalize entropy by sequence length")
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/analyse_attention.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    attack_items = args.attack_dir_path
    attack_items = attack_items.split('/')

    # Load model
    model = select_model(args.model_name, model_path=args.model_path)
    device = torch.device('cpu')
    model.to(device)
    model.eval()

    # Load and filter data
    o_sen, a_sen, o_pred, a_pred, labels = select_attacked_data(args.attack_dir_path, args.part)
    success, unsuccess = Attacker.get_success_and_unsuccess_attacks(o_sen, a_sen, o_pred, a_pred, labels)

    analyzer = AttentionAnalyzer(model=model)
    out_str = ''

    # Kl Divergence calculation
    if not args.KL_off:
        success_kls, success_lengths = analyzer.attn_kl_div_all(success['o_sens'], success['a_sens'], layer=args.layer)
        unsuccess_kls, unsuccess_lengths = analyzer.attn_kl_div_all(unsuccess['o_sens'], unsuccess['a_sens'], layer=args.layer)

        out_str += f'\nSuccessful attacks KL-div\t{mean(success_kls)}+-{stdev(success_kls)}'
        try:
            out_str += f'\nUnsuccessful attacks KL-div\t{mean(unsuccess_kls)}+-{stdev(unsuccess_kls)}\n\n'
        except:
            out_str += '\nNo Unsuccessful Attacks\n\n'

    # Entropy calculation of attention distribution
    if not args.entropy_off:
        suc_ents_o, suc_ents_a, suc_l_os, suc_l_as = analyzer.entropy_all(success['o_sens'], success['a_sens'], layer=args.layer, align=args.align)
        unsuc_ents_o, unsuc_ents_a, unsuc_l_os, unsuc_l_as = analyzer.entropy_all(unsuccess['o_sens'], unsuccess['a_sens'], layer=args.layer, align=args.align)
        
        out_str += f'\nSuccessful-Original attn entropy:\t{mean(suc_ents_o)}+-{stdev(suc_ents_o)}\t\tLength:\t{mean(suc_l_os)}+-{stdev(suc_l_os)}'
        out_str += f'\nSuccessful-Attacked attn entropy:\t{mean(suc_ents_a)}+-{stdev(suc_ents_a)}\t\tLength:\t{mean(suc_l_as)}+-{stdev(suc_l_as)}'
        try:
            out_str += f'\nUnsuccessful-Original attn entropy:\t{mean(unsuc_ents_o)}+-{stdev(unsuc_ents_o)}\t\tLength:\t{mean(unsuc_l_os)}+-{stdev(unsuc_l_os)}'
            out_str += f'\nUnsuccessful-Attacked attn entropy:\t{mean(unsuc_ents_a)}+-{stdev(unsuc_ents_a)}\t\tLength:\t{mean(unsuc_l_as)}+-{stdev(unsuc_l_as)}'
        except:
            out_str += '\nNo Unsuccessful Attacks\n\n'
    
    # Embedding Change
    if not args.emb_dist_off:
        suc_diffs = analyzer.emb_change_all(success['o_sens'], success['a_sens'], layer=args.layer, dist=args.dist)
        unsuc_diffs = analyzer.emb_change_all(unsuccess['o_sens'], unsuccess['a_sens'], layer=args.layer, dist=args.dist)

        out_str += f'\nSuccessful attacks embedding change ({args.dist})\t{mean(suc_diffs)}+-{stdev(suc_diffs)}'
        try:
            out_str += f'\nUnsuccessful attacks embedding change ({args.dist})\t{mean(unsuc_diffs)}+-{stdev(unsuc_diffs)}'
        except:
            out_str += '\nNo Unsuccessful Attacks\n\n'

    # Entropy of output
    if not args.out_entropy_off:
        suc_ent = analyzer.out_entropy_all(success['o_sens'])
        unsuc_ent = analyzer.out_entropy_all(unsuccess['o_sens'])
        labels = [1]*len(suc_ent) + [0]*len(unsuc_ent)
        attack_recalls, rets = Retention.retention_curve_frac_positive(suc_ent+unsuc_ent, labels)
        opt2_rec, _ = Retention.retention_curve_frac_positive(labels, labels)
        opt1_rec, _ = Retention.retention_curve_frac_positive([1-l for l in labels], labels)

        # Create retention plot
        out_path_part = '_'.join(attack_items[-4:])
        out_path = f'{args.out_path_plot}/out_entropy_{out_path_part}.png'

        plt.plot(rets, attack_recalls, label=f'{attack_items[-1]} output entropy')
        plt.plot(rets, rets, label='No correlation', linestyle='dashed', color='red', linewidth=0.5)
        plt.plot(rets, opt1_rec, label='Limit 1', linestyle='dashed', color='purple', linewidth=0.5)
        plt.plot(rets, opt2_rec, label='Limit 2', linestyle='dashed', color='purple', linewidth=0.5)
        plt.ylabel('Attackable Samples Recall Rate')
        plt.xlabel('Retention Fraction by lowest output entropy')
        plt.legend()
        plt.savefig(out_path, bbox_inches='tight')
        plt.clf()

        out_str += f'\nSuccessful Original attacks output entropy \t{mean(suc_ent)}+-{stdev(suc_ent)}'
        try:
            out_str += f'\nUnsuccessful Original attacks output entropy \t{mean(unsuc_ent)}+-{stdev(unsuc_ent)}'
        except:
            out_str += '\nNo Unsuccessful Attacks\n\n'
    
    # Entropy of attention distribution with retention plots
    if not args.attn_entropy_off:
        layers = [1,6,12]
        for l in layers:
            suc_ent, suc_lens = analyzer.entropy_all(success['o_sens'], num_heads=1, layer=l)
            unsuc_ent, unsuc_lens = analyzer.entropy_all(unsuccess['o_sens'], num_heads=1, layer=l)
            labels = [1]*len(suc_ent) + [0]*len(unsuc_ent)
            ents = suc_ent+unsuc_ent
            if args.norm_ent:
                # normalize entropy by sequence length
                lens = suc_lens + unsuc_lens
                ents = [e/math.log(l) for e,l in zip(ents, lens)]
            attack_recalls, rets = Retention.retention_curve_frac_positive(ents, labels)

            # Create retention plot
            plt.plot(rets, attack_recalls, label=f'layer {l} attn entropy')

        opt2_rec, _ = Retention.retention_curve_frac_positive(labels, labels)
        opt1_rec, _ = Retention.retention_curve_frac_positive([1-l for l in labels], labels)
        out_path_part = '_'.join(attack_items[-4:])
        out_path = f'{args.out_path_plot}/attn_entropy_{out_path_part}.png'
        if args.norm_ent:
            out_path = f'{args.out_path_plot}/attn_entropy_norm_{out_path_part}.png'
        plt.plot(rets, rets, label='No correlation', linestyle='dashed', color='red', linewidth=0.5)
        plt.plot(rets, opt1_rec, label='Limit 1', linestyle='dashed', color='purple', linewidth=0.5)
        plt.plot(rets, opt2_rec, label='Limit 2', linestyle='dashed', color='purple', linewidth=0.5)
        plt.ylabel('Attackable Samples Recall Rate')
        plt.xlabel('Retention Fraction by lowest attn entropy')
        plt.legend()
        plt.savefig(out_path, bbox_inches='tight')
        plt.clf()


    

    print(out_str)

    # log
    if args.log_dir != 'none':
        attack_items = '_'.join(attack_items[-4:])
        out_path = f'{args.log_dir}/{attack_items}_layer{args.layer}.txt'
        with open(out_path, 'w') as f:
            f.write(out_str)

    