from torch_geometric.data import Data, Batch
import torch
import pandas as pd
from utils.smiles2graph import smiles2graph
from torch_geometric.data import DataLoader
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import random
import copy

def graphdata_to_pygdata(features : list, labels : list | None) -> list:

    if labels is not None:
        assert len(features) == len(labels)
    data = []
    for i in range(len(features)):
        graph_data = features[i]
        x = torch.tensor(graph_data.node_features, dtype=torch.float)

        edge_index = torch.tensor(graph_data.edge_index, dtype=torch.long)
        edge_attr = None
        if graph_data.edge_features is not None:
            edge_attr = torch.tensor(graph_data.edge_features, dtype=torch.float)

        if labels is not None:
            data.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=labels[i]))
        else:
            data.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
    return data

def graphdata_to_pygdata_PI(features : list, labels : list) -> list:
    assert len(features) == len(labels)
    data = []
    for i in range(len(features)):
        graph_data = features[i]
        x = torch.tensor(graph_data.node_features, dtype=torch.float)

        edge_index = torch.tensor(graph_data.edge_index, dtype=torch.long)

        edge_attr = None
        if graph_data.edge_features is not None:
            edge_attr = torch.tensor(graph_data.edge_features, dtype=torch.float)
        y = torch.tensor(labels[i], dtype=torch.float).unsqueeze(0)
        data.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return data

class MetalLigandDataset(torch.utils.data.Dataset):
    def __init__(self, graph_data_list, metal_features_list):
        self.graph_data_list = graph_data_list
        self.metal_features_list = metal_features_list

    def __len__(self):
        return len(self.graph_data_list)

    def __getitem__(self, idx):
        return self.graph_data_list[idx], self.metal_features_list[idx]

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def test_dataloader(path, bs, metal='deltaG_Ni_4'):
    metal_descriptors = {
        'deltaG_Ni_4': {
            'Ionic_Radius': 0.55,
            'Atomic_Number': 28,
            'MW': 58.69,
            'coord_num': 4,
            'electronegativity': 1.91,
            'FirstIE': 7.63,
            'SecondIE': 18.16,
            'ThirdIE': 35.19,
        },
        'deltaG_Co_4': {
            'Ionic_Radius': 0.58,
            'Atomic_Number': 27,
            'MW': 58.93,
            'coord_num': 4,
            'electronegativity': 1.88,
            'FirstIE': 7.88,
            'SecondIE': 17.08,
            'ThirdIE': 33.50,
        },
        'deltaG_Mn_4': {
            'Ionic_Radius': 0.66,
            'Atomic_Number': 25,
            'MW': 54.94,
            'coord_num': 4,
            'electronegativity': 1.55,
            'FirstIE': 7.43,
            'SecondIE': 15.63,
            'ThirdIE': 33.66,
        },
        'deltaG_Ni_8': {
            'Ionic_Radius': 0.69,
            'Atomic_Number': 28,
            'MW': 58.69,
            'coord_num': 8,
            'electronegativity': 1.91,
            'FirstIE': 7.63,
            'SecondIE': 18.16,
            'ThirdIE': 35.19,
        },
        'deltaG_Co_8': {
            'Ionic_Radius': 0.745,
            'Atomic_Number': 27,
            'MW': 58.93,
            'coord_num': 8,
            'electronegativity': 1.88,
            'FirstIE': 7.88,
            'SecondIE': 17.08,
            'ThirdIE': 33.50,
        },
        'deltaG_Mn_8': {
            'Ionic_Radius': 0.85,
            'Atomic_Number': 25,
            'MW': 54.94,
            'coord_num': 8,
            'electronegativity': 1.55,
            'FirstIE': 7.43,
            'SecondIE': 15.63,
            'ThirdIE': 33.66,
        }
    }
    df = pd.read_csv(path) if isinstance(path, str) else copy.deepcopy(path)
    for k in metal_descriptors[metal]:
        df[k] = metal_descriptors[metal][k]
    smi_list = df['smiles'].tolist()
    compound_list = df['compound'].tolist()
    ion_radius = df['Ionic_Radius'].tolist()
    atom_num = df['Atomic_Number'].tolist()
    electronegativity = df['electronegativity'].tolist()
    mw = df['MW'].tolist()
    coord_num = df['coord_num'].tolist()
    FirstIE = df['FirstIE'].tolist()
    SecondIE = df['SecondIE'].tolist()
    ThirdIE = df['ThirdIE'].tolist()
    X = smiles2graph(smi_list)
    data_list = graphdata_to_pygdata(X, None)
    metal_features_list = []
    for i in range(len(smi_list)):
        data_list[i].smiles = smi_list[i]
        data_list[i].metal = metal
        data_list[i].compound = compound_list[i]
        metal_features_list.append(torch.tensor([ion_radius[i], atom_num[i], electronegativity[i], mw[i],
                                                 coord_num[i], FirstIE[i], SecondIE[i], ThirdIE[i]], dtype=torch.float))
    dataset = MetalLigandDataset(data_list, metal_features_list)

    def collate_fn(batch):
        graph_data_list, metal_features_list = zip(*batch)
        batch_graph_data = Batch.from_data_list(graph_data_list)
        batch_metal_features = torch.stack(metal_features_list, dim=0)
        return batch_graph_data, batch_metal_features

    dataloader = DataLoader(dataset, batch_size=bs, collate_fn=collate_fn, shuffle=False, worker_init_fn=seed_worker)
    return dataloader

def internal_dataloader(df, shuffle, bs, scaler):
    target_columns = ["A_E_HOMO (a.u.)", "A_E_LUMO (a.u.)", "A_Electric Density (g/cm^3)",
                      "A_Potential Minimal value (kcal/mol)", "Vertical IP", "Mulliken electronegativity",
                      "Chemical potential", "Hardness", "Softness", "Nucleophilicity index"]
    smi_list = df['smiles'].tolist()
    compound_list = df['compound'].tolist()
    metal_list = df['Metal'].tolist()
    deltaG = df['delta_G'].tolist()
    ion_radius = df['Ionic_Radius'].tolist()
    atom_num = df['Atomic_Number'].tolist()
    electronegativity = df['electronegativity'].tolist()
    mw = df['MW'].tolist()
    coord_num = df['coord_num'].tolist()
    FirstIE = df['FirstIE'].tolist()
    SecondIE = df['SecondIE'].tolist()
    ThirdIE = df['ThirdIE'].tolist()
    X = smiles2graph(smi_list)
    if scaler is not None:
        targets = df[target_columns].values
        targets_normalized = scaler.transform(targets)
        f_list = targets_normalized.tolist()
    data_list = graphdata_to_pygdata(X, deltaG)
    metal_features_list = []
    for i in range(len(smi_list)):
        data_list[i].smiles = smi_list[i]
        data_list[i].compound = compound_list[i]
        data_list[i].metal = metal_list[i]
        if scaler is not None:
            data_list[i].f = torch.tensor(f_list[i], dtype=torch.float).unsqueeze(0)
        metal_features_list.append(torch.tensor([ion_radius[i], atom_num[i], electronegativity[i], mw[i],
                                                 coord_num[i], FirstIE[i], SecondIE[i], ThirdIE[i]], dtype=torch.float))
    dataset = MetalLigandDataset(data_list, metal_features_list)

    def collate_fn(batch):
        graph_data_list, metal_features_list = zip(*batch)
        batch_graph_data = Batch.from_data_list(graph_data_list)
        batch_metal_features = torch.stack(metal_features_list, dim=0)
        return batch_graph_data, batch_metal_features

    dataloader = DataLoader(dataset, batch_size=bs, collate_fn=collate_fn, shuffle=shuffle, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(20))
    return dataloader

def prepare_dataloader(path, shuffle, bs, scaler=None):
    df = pd.read_csv(path) if isinstance(path, str) else copy.deepcopy(path)
    return internal_dataloader(df, shuffle, bs, scaler)

def prepare_dataloader_PI(path, shuffle, bs, scaler, fit, test_path=None, train_path=None, val_path=None):
    target_columns = ["A_E_HOMO (a.u.)", "A_E_LUMO (a.u.)", "A_Electric Density (g/cm^3)",
                      "A_Potential Minimal value (kcal/mol)", "Vertical IP", "Mulliken electronegativity",
                      "Chemical potential", "Hardness", "Softness", "Nucleophilicity index"]
    metal_features = ['Ionic_Radius', 'Atomic_Number', 'electronegativity', 'MW', 'coord_num', 'FirstIE', 'SecondIE', 'ThirdIE']
    df = pd.read_csv(path)
    df = df.dropna(subset=target_columns).reset_index()
    if test_path is not None:
        df_test = pd.read_csv(test_path) if isinstance(test_path, str) else copy.deepcopy(test_path)
        df = df[~df['compound'].isin(df_test['compound'])]#.reset_index()
    if val_path is not None:
        df_val = pd.read_csv(val_path) if isinstance(val_path, str) else copy.deepcopy(val_path)
        df = df[~df['compound'].isin(df_val['compound'])].reset_index()
    if train_path is not None:
        df_train = pd.read_csv(train_path) if isinstance(train_path, str) else copy.deepcopy(train_path)
        df = df[df['compound'].isin(df_train['compound'])].reset_index()
    smi_list = df['smiles'].tolist()

    X = smiles2graph(smi_list)
    targets = df[target_columns].values
    if fit:
        targets_normalized = scaler.fit_transform(targets)
    else:
        targets_normalized = scaler.transform(targets)
    df_normalized = df.copy()
    df_normalized[target_columns] = targets_normalized

    labels = df_normalized[target_columns].values.tolist()
    data_list = graphdata_to_pygdata_PI(X, labels)
    return DataLoader(data_list, batch_size=bs, shuffle=shuffle, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(20))


def prepare_PI_model(path, scaler):
    target_columns = ["A_E_HOMO (a.u.)", "A_E_LUMO (a.u.)", "A_Electric Density (g/cm^3)",
                      "A_Potential Minimal value (kcal/mol)", "Vertical IP", "Mulliken electronegativity",
                      "Chemical potential", "Hardness", "Softness", "Nucleophilicity index"]
    metal_features = ['Ionic_Radius', 'Atomic_Number', 'electronegativity', 'MW', 'coord_num', 'FirstIE', 'SecondIE',
                      'ThirdIE']
    df = pd.read_csv(path) if isinstance(path, str) else copy.deepcopy(path)
    df = df.dropna(subset=target_columns).reset_index()
    targets = df[target_columns].values
    targets_normalized = scaler.transform(targets)
    df_scaled = pd.DataFrame(targets_normalized, columns=target_columns)
    X_train = pd.concat([df_scaled, df[metal_features]], axis=1)
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=20)
    model.fit(X_train.values, df['delta_G'])
    return model

