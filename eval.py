import torch
import torch.nn as nn
import sys
import os
import argparse
from src.tools.tools import get_default_device
from src.models.model_selector import select_model
from src.data.select_data import select_data
from src.training.trainer import Trainer

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--model_path', type=str, required=True, help='Specify path to saved model')
    commandLineParser.add_argument('--model_name', type=str, required=True, help='e.g. bert-base-cased')
    commandLineParser.add_argument('--data_dir_path', type=str, required=True, help='path to data directory')
    commandLineParser.add_argument('--bs', type=int, default=8, help="Specify batch size")
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    if args.force_cpu == 'yes':
        device = torch.device('cpu')
    else:
        device = get_default_device()

    # Load model
    model = select_model(args.model_name, model_path=args.model_path)
    model.to(device)

    # Load the test data
    sentences, labels = select_data(data_dir_path=args.data_dir_path, part='test')
    _, dl = Trainer.split_and_dl(model, sentences, labels, val=1.0, bs=args.bs)

    # Criterion
    criterion = nn.CrossEntropyLoss().to(device)

    # Evaluate
    acc = Trainer.eval(dl, criterion, device)
    print('Accuracy', acc)
