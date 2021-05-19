from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import RDConfig
from torch_geometric.data import Data
from torch_geometric import data as DATA
import numpy as np
import sys
import os
import torch
import torch_geometric
from models.dta.GCNNet import GCNNet
import networkx as nx

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))


def morgan_fingerprint(smiles, fp_length, fp_radius):
    if smiles is None:
        return None

    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        return None

    return AllChem.GetMorganFingerprintAsBitVect(
        molecule,
        fp_radius,
        fp_length
    )


def numpy_morgan_fingerprint(smiles, fp_length, fp_radius):
    fingerprint = morgan_fingerprint(smiles, fp_length, fp_radius)

    if fingerprint is None:
        return np.zeros((fp_length,))

    arr = np.zeros((1,))

    DataStructs.ConvertToNumpyArray(fingerprint, arr)

    return arr


def atom_valences(atom_types):
    periodic_table = Chem.GetPeriodicTable()
    return [
        max(list(periodic_table.GetValenceList(atom_type)))
        for atom_type in atom_types
    ]


def check_molecule_validity(mol, transform):
    if type(mol) == torch_geometric.data.Data:
        mol = transform(mol)

    return Chem.SanitizeMol(mol, catchErrors=True) == Chem.SANITIZE_NONE


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def mol_to_dta_pyg(mol):
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    data = Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index).transpose(1, 0), )
    data.batch = torch.zeros(len(features)).long()

    return data


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index


def get_graphdta_dgn(model_file_name='models/dta/model_GCNNet_davis.model'):
    m = GCNNet()
    m.load_state_dict(torch.load(model_file_name))
    m.eval()
    return m
