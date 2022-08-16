'''
Generate attack outputs
'''

import sys
import os
import argparse
from src.tools.tools import set_seeds
from src.models.model_selector import select_model
from src.data.select_data import select_data
from src.attack.generate_attack import Attacker

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--model_path', type=str, required=True, help='Specify path to saved model')
    commandLineParser.add_argument('--model_name', type=str, required=True, help='e.g. bert-base-uncased')
    commandLineParser.add_argument('--data_dir_path', type=str, required=True, help='e.g. src/data/data_files/imdb')
    commandLineParser.add_argument('--out_dir_path', type=str, required=True, help='e.g. src/data/data_files/imdb/attacks/bert/pwws')
    commandLineParser.add_argument('--attack_recipe', type=str, required=True, help='e.g. pwws')
    commandLineParser.add_argument('--part', type=str, default='test', help="part of data")
    commandLineParser.add_argument('--lev_dist_constraint', type=float, default=-1.0, help="threshold for specific attack")
    commandLineParser.add_argument('--use_constraint', type=float, default=-1.0, help="threshold for specific attack")
    commandLineParser.add_argument('--seed', type=int, default=1, help="Specify seed")
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    set_seeds(args.seed)
    if args.lev_dist_constraint == -1.0:
        lev_dist_constraint = None
    if args.use_constraint == -1.0:
        use_constraint = None

    # Load model
    model = select_model(args.model_name, model_path=args.model_path)

    # Load the test data
    sentences, labels = select_data(data_dir_path=args.data_dir_path, part=args.part)

    # Attack
    attacker = Attacker(model, attack_recipe=args.attack_recipe,
                         use_constraint=use_constraint, lev_dist_constraint=lev_dist_constraint)
    attacked_sentences = []
    original_predictions = []
    attacked_predictions = []
    for i, (sentence, label) in enumerate(zip(sentences,  labels)):
        print(f'On {i}/{len(sentences)}')
        attacked_sentence, orig_pred_class, attacked_pred_class = attacker.attack_sentence(sentence, label)
        attacked_predictions.append(attacked_sentence+'\n')
        original_predictions.append(str(orig_pred_class)+'\n')
        attacked_predictions.append(str(attacked_pred_class)+'\n')
        print(f'Original: {sentence}')
        print(f'Attacked: {attacked_sentence}')
        print(f'Original prediction: {orig_pred_class}')
        print(f'Attacked prediction: {attacked_pred_class}')
        print()
    
    # save
    with open(f'{args.data_dir_path}/{args.part}_attacked_sentences.txt', 'w') as f:
        f.writelines(attacked_sentences)
    with open(f'{args.data_dir_path}/{args.part}_original_predictions.txt', 'w') as f:
        f.writelines(original_predictions)
    with open(f'{args.data_dir_path}/{args.part}_attacked_predictions.txt', 'w') as f:
        f.writelines(attacked_predictions)
