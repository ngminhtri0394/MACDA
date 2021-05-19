import os

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch_geometric.data import DataLoader as GeoDataLoader

from mol_utils.TestbedDataset import TestbedDataset


def pad(sample, n_pad):
    sample.x = F.pad(sample.x, (0, n_pad), "constant", 0)
    return sample


def seq_cat(prot):
    max_seq_len = 1000
    seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
    seq_dict_len = len(seq_dict)
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    x = np.asarray(x)
    x = torch.LongTensor([x])
    return x


def preprocess(dataset_name, args):
    return _PREPROCESS[dataset_name.lower()](args)


def _preprocess_davis_graphdta(args):
    processed_data_file_train = 'data/davis/processed/' + args.preprocessfile + '.pt'

    if not os.path.isfile(processed_data_file_train):
        print('please run create_data.py to prepare data in pytorch format!')
        exit()
    else:
        train_data = TestbedDataset(root='data/davis/', dataset=args.preprocessfile)

        df = pd.read_csv('data/davis/' + args.preprocessfile + '.csv')
        train_drugs, train_prots, train_prots_seq, train_Y = list(df['compound_iso_smiles']), list(
            df['target_name']), list(
            df['target_sequence']), list(df['affinity'])

        # make data PyTorch mini-batch processing ready
        train_loader = GeoDataLoader(train_data, batch_size=1, shuffle=False)
        return train_loader, train_drugs, train_Y

_PREPROCESS = {
    'davis_graphdta': _preprocess_davis_graphdta
}
