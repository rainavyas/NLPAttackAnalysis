import sys
import os
import argparse
from .data_utils import load_data

def split_dict_and_save(dict_object_lst, data_dir_path, part='test'):
    sentences = [d['text']+'\n' for d in dict_object_lst]
    labels = [d['label']+'\n' for d in dict_object_lst]

    with open(f'{data_dir_path}/{part}_sentences.txt', 'w') as f:
        f.writelines(sentences)

    with open(f'{data_dir_path}/{part}_labels.txt', 'w') as f:
        f.writelines(labels)


if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--data_name', type=str, required=True, help='e.g. imdb')
    commandLineParser.add_argument('--out_dir', type=str, required=True, help='e.g. ../imdb')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/preprocess.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    train, _, test = load_data(args.data_name)
    split_dict_and_save(train, args.out_dir, part='train')
    split_dict_and_save(test, args.out_dir, part='test')