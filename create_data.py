import os
import sys

import numpy as np
import pandas as pd

import mol_utils
from mol_utils import TestbedDataset

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

datafile = sys.argv[1]
compound_iso_smiles = []
df = pd.read_csv('data/davis/' + datafile + '.csv')
compound_iso_smiles += list(df['compound_iso_smiles'])
compound_iso_smiles = set(compound_iso_smiles)
smile_graph = {}
for smile in compound_iso_smiles:
    g = mol_utils.smile_to_graph(smile)
    smile_graph[smile] = g

datasets = ['davis']
# convert to PyTorch data format
for dataset in datasets:
    processed_data_file_train = 'data/processed/' + dataset + datafile + '.pt'
    if (not os.path.isfile(processed_data_file_train)):
        df = pd.read_csv('data/davis/' + datafile + '.csv')
        train_drugs, train_prots, train_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
            df['affinity'])
        XT = [t for t in train_prots]
        train_drugs, train_prots, train_Y = np.asarray(train_drugs), np.asarray(XT), np.asarray(train_Y)

        # make data PyTorch Geometric ready
        print('preparing ', dataset + '_train.pt in pytorch format!')
        train_data = TestbedDataset(root='data', dataset=dataset + datafile, xd=train_drugs, xt=train_prots,
                                    y=train_Y, smile_graph=smile_graph)
        print(processed_data_file_train, ' and have been created')
    else:
        print(processed_data_file_train, 'are already created')
