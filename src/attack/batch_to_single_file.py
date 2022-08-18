'''
Transforms the output of attack_batch.py files into a single correctly ordered set of files:
    - original_sentences.txt
    - original_labels.txt
    - attacked_sentences.txt
    - original_predictions.txt
    - attacked_predictions.txt
'''

import sys
import os
import argparse

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--batch_dir_path', type=str, required=True, help='e.g. src/data/data_files/imdb/attacks/bert/pwws/batch_output')
    commandLineParser.add_argument('--out_dir_path', type=str, required=True, help='e.g. src/data/data_files/imdb/attacks/bert/pwws')
    commandLineParser.add_argument('--part', type=str, default='test', help="part of data")
    commandLineParser.add_argument('--num_files', type=int, default=22, help="number of batch files in dir")
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/batch_to_single_file.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # Load all files
    original_sentences = []
    original_labels = []
    attacked_sentences = []
    attacked_predictions = []
    original_predictions = []
    for i in range(args.num_files):
        try:
            with open(f'{args.batch_dir_path}/{i}_{args.part}_original_sentences.txt', 'r') as f:
                vals = f.readlines()
            original_sentences += vals
        except:
            continue
        
        with open(f'{args.batch_dir_path}/{i}_{args.part}_original_labels.txt', 'r') as f:
            vals = f.readlines()
        original_labels += vals
        with open(f'{args.batch_dir_path}/{i}_{args.part}_attacked_sentences.txt', 'r') as f:
            vals = f.readlines()
        attacked_sentences += vals
        with open(f'{args.batch_dir_path}/{i}_{args.part}_attacked_predictions.txt', 'r') as f:
            vals = f.readlines()
        attacked_predictions += vals
        with open(f'{args.batch_dir_path}/{i}_{args.part}_original_predictions.txt', 'r') as f:
            vals = f.readlines()
        original_predictions += vals
    
    # save
    with open(f'{args.out_dir_path}/{args.part}_original_sentences.txt', 'w') as f:
        f.writelines(original_sentences)
    with open(f'{args.out_dir_path}/{args.part}_original_labels.txt', 'w') as f:
        f.writelines(original_labels)
    with open(f'{args.out_dir_path}/{args.part}_attacked_sentences.txt', 'w') as f:
        f.writelines(attacked_sentences)
    with open(f'{args.out_dir_path}/{args.part}_original_predictions.txt', 'w') as f:
        f.writelines(original_predictions)
    with open(f'{args.out_dir_path}/{args.part}_attacked_predictions.txt', 'w') as f:
        f.writelines(attacked_predictions)
    


    