import sys
import os
import argparse
from src.attention.attention import AttentionAnalyzer
from src.models.model_selector import select_model
import torch

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--model_path', type=str, required=True, help='path of saved model')
    commandLineParser.add_argument('--model_name', type=str, required=True, help='e.g. bert-base-uncased')
    commandLineParser.add_argument('--original_data', type=str, required=True, help='path to original data file')
    commandLineParser.add_argument('--attacked_data', type=str, required=True, help="path to attacked data file")
    commandLineParser.add_argument('--out_dir', type=str, required=True, help="Dir to save generated histograms")
    commandLineParser.add_argument('--num_samples', type=int, default=20, help="Specify number of data samples to use")
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
    
    # temp to generate some plots
    original_sentence = "I love sci-fi and am willing to put up with a lot. Sci-fi movies/TV are usually underfunded, under-appreciated and misunderstood. I tried to like this, I really did, but it is to good TV sci-fi as Babylon 5 is to Star Trek (the original). Silly prosthetics, cheap cardboard sets, stilted dialogues, CG that doesn't match the background, and painfully one-dimensional characters cannot be overcome with a 'sci-fi' setting. (I'm sure there are those of you out there who think Babylon 5 is good sci-fi TV. It's not. It's clichéd and uninspiring.) While US viewers might like emotion and character development, sci-fi is a genre that does not take itself seriously (cf. Star Trek). It may treat important issues, yet not as a serious philosophy. It's really difficult to care about the characters here as they are not simply foolish, just missing a spark of life. Their actions and reactions are wooden and predictable, often painful to watch. The makers of Earth KNOW it's rubbish as they have to always say 'Gene Roddenberry's Earth...' otherwise people would not continue watching. Roddenberry's ashes must be turning in their orbit as this dull, cheap, poorly edited (watching it without advert breaks really brings this home) trudging Trabant of a show lumbers into space. Spoiler. So, kill off a main character. And then bring him back as another actor. Jeeez! Dallas all over again."
    attack_sentence = "I adore sci-fi and am willing to enjoy with a lot. Sci-fi movies/TV are usually underfunded, under-appreciated and misunderstood. I wanted to like this, I really did, but it is to good TV sci-fi as Babylon 5 is to Star Trek (the original). Funny prosthetics, cheap cardboard sets, funny dialogues, CG that doesn't match the background, and oddly one-dimensional characters cannot be overcome with a 'sci-fi' setting. (I'm sure there are those of you out there who think Babylon 5 is good sci-fi TV. It's not. It's clichéd and uninspiring.) While US viewers might like emotion and character development, sci-fi is a genre that does not take itself seriously (cf. Star Trek). It may treat important issues, yet not as a serious philosophy. It's really difficult to care about the characters here as they are not simply foolish, just missing a spark of life. Their actions and reactions are wooden and predictable, often painful to watch. The makers of Earth KNOW it's rubbish as they have to always say 'Gene Roddenberry's Earth...' otherwise people would not continue watching. Roddenberry's ashes must be turning in their orbit as this dull, cheap, poorly edited (watching it without advert breaks really brings this home) trudging Trabant of a show lumbers into space. Spoiler. So, kill off a main character. And then bring him back as another actor. Jeeez! Dallas all over again."


    analyzer = AttentionAnalyzer(model=model)
    analyzer.visualize_attack(original_sentence, attack_sentence, 'temp', layer=args.layer)