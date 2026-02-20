import deepchem as dc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def smiles2graph(smi : list) -> list:
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    X = featurizer.featurize(smi)
    return X



def weights_to_colormap(weights):

    colors_rgb = [
        (26, 49, 139),    
        (73, 108, 206),
        (130, 170, 231),
        (185, 210, 243),
        (230, 240, 254), 
        (249, 219, 229), 
        (247, 166, 191),
        (228, 107, 144),
        (192, 63, 103),
        (154, 19, 61)  
    ]
    colors_normalized = [(r/255, g/255, b/255) for r, g, b in colors_rgb]
    custom_cmap = LinearSegmentedColormap.from_list("custom_blue_red", colors_normalized)

    weights = np.array(weights)
    if np.max(weights) == np.min(weights):
        norm_weights = np.zeros_like(weights, dtype=float)
    else:
        norm_weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
    rgb_tuples = [tuple(row[:3]) for row in custom_cmap(norm_weights)]

    return rgb_tuples
