def select_data(data_dir_path='./src/data/imdb', part='train'):

    with open(f'{data_dir_path}/{part}_sentences.txt') as f:
        sentences = f.readlines()
    sentences = [s.rstrip('\n') for s in sentences]

    with open(f'{data_dir_path}/{part}_labels.txt') as f:
        labels = f.readlines()
    labels = [int(l.rstrip('\n')) for l in labels]

    return sentences, labels