def select_attacked_data(attack_data_dir_path='src/data/data_files/imdb/attacks/bert/pwws', part='test'):

    with open(f'{attack_data_dir_path}/{part}_original_sentences.txt') as f:
        o_sen = f.readlines()
    o_sen = [s.rstrip('\n') for s in o_sen]

    with open(f'{attack_data_dir_path}/{part}_original_labels.txt') as f:
        labels = f.readlines()
    labels = [int(l.rstrip('\n')) for l in labels]

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
