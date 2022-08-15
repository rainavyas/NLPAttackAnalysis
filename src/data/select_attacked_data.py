from .select_data import select_data

def select_attacked_data(data_dir_path='./src/data/data_files/imdb',
                             attack_data_dir_path='src/data/data_files/imdb/attacks/bert/pwws', part='test'):

    o_sen, labels = select_data(data_dir_path=data_dir_path, part=part)

    with open(f'{attack_data_dir_path}/{part}_attacked_sentences.txt') as f:
        a_sen = f.readlines()
    a_sen = [s.rstrip('\n') for s in a_sen]

    with open(f'{attack_data_dir_path}/{part}_original_predictions.txt') as f:
        o_pred = f.readlines()
    o_pred = [int(l.rstrip('\n')) for l in o_pred]

    with open(f'{attack_data_dir_path}/{part}_attacked_predictions.txt') as f:
        a_pred = f.readlines()
    a_pred = [int(l.rstrip('\n')) for l in a_pred]

    return o_sen, a_sen, o_pred, a_pred, labels
