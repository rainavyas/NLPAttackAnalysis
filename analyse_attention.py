import sys
import os
import argparse

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--model_path', type=str, required=True, help='path of saved model')
    commandLineParser.add_argument('--original_data', type=str, required=True, help='path to original data file')
    commandLineParser.add_argument('--attacked_data', type=str, required=True, help="path to attacked data file")
    commandLineParser.add_argument('--out_dir', type=str, help="Dir to save generated histograms")
    commandLineParser.add_argument('--num_samples', type=int, default=20, help="Specify number of data samples to use")
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/analyse_attention.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')