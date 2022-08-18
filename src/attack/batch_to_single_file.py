'''
Transforms the output of attack_batch.py files into a single correctly ordered set of files:
    - attacked_sentences.txt
    - original_predictions.txt
    - attacked_predictions.txt

-> note: 'none' is placed for samples with no predictions
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
    commandLineParser.add_argument('--batch_size', type=int, default=50, help="number of samples in file")
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/batch_to_single_file.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # Load all files
    attacked_sentences = []
    attacked_predictions = []
    original_predictions = []
    for i in range(args.num_files):

        try:
            with open(f'{args.batch_dir_path}/{i}_{args.part}_attacked_sentences.txt', 'r') as f:
                vals = f.readlines()
            vals = [s.rstrip('\n') for s in vals]
            attacked_sentences += vals

            with open(f'{args.batch_dir_path}/{i}_{args.part}_attacked_predictions.txt', 'r') as f:
                vals = f.readlines()
            vals = [s.rstrip('\n') for s in vals]
            attacked_predictions += vals

            with open(f'{args.batch_dir_path}/{i}_{args.part}_original_predictions.txt', 'r') as f:
                vals = f.readlines()
            vals = [s.rstrip('\n') for s in vals]
            original_predictions += vals
        except:
            attacked_sentences += ['none']*args.batch_size
            attacked_predictions += ['-1']*args.batch_size
            original_predictions += ['-1']*args.batch_size
    
    # save
    with open(f'{args.out_dir_path}/{args.part}_attacked_sentences.txt', 'w') as f:
        f.writelines(attacked_sentences)
    with open(f'{args.out_dir_path}/{args.part}_original_predictions.txt', 'w') as f:
        f.writelines(original_predictions)
    with open(f'{args.out_dir_path}/{args.part}_attacked_predictions.txt', 'w') as f:
        f.writelines(attacked_predictions)
    


    