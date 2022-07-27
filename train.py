import torch
import torch.nn as nn
import sys
import os
import argparse
from src.tools.tools import get_default_device, set_seeds
from src.models.model_selector import select_model
from src.data.select_data import select_data
from src.training.trainer import Trainer

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--out_dir', type=str, required=True, help='Specify dir to save model')
    commandLineParser.add_argument('--model_name', type=str, required=True, help='e.g. bert-base-cased')
    commandLineParser.add_argument('--data_dir_path', type=str, required=True, help='path to data directory')
    commandLineParser.add_argument('--bs', type=int, default=8, help="Specify batch size")
    commandLineParser.add_argument('--epochs', type=int, default=10, help="Specify max epochs")
    commandLineParser.add_argument('--lr', type=float, default=0.00001, help="Specify learning rate")
    commandLineParser.add_argument('--sch', type=int, default=[3], nargs='+', help="Specify scheduler cycle, e.g. 10 100 1000")
    commandLineParser.add_argument('--seed', type=int, default=1, help="Specify seed")
    commandLineParser.add_argument('--force_cpu', type=str, choices=['yes', 'no'], default='no', help='force cpu use')
    args = commandLineParser.parse_args()

    set_seeds(args.seed)
    out_file = f'{args.out_dir}/{args.model_name}_{args.data_name}_seed{args.seed}.th'

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    if args.force_cpu == 'yes':
        device = torch.device('cpu')
    else:
        device = get_default_device()

    # Load the training data
    sentences, labels = select_data(data_dir_path=args.data_dir_path, part='train')

    # Initialise model
    model = select_model(args.model_name)
    model.to(device)

    # Define learning objects
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.sch)
    criterion = nn.CrossEntropyLoss().to(device)

    # Train
    trainer = Trainer(device, model, optimizer, criterion, scheduler)
    trainer.train_process(sentences, labels, out_file, max_epochs=args.epochs, bs=args.bs)