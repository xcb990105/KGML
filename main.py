import sys
from models.delta_G.gat_deltaG_prediction import GATDeltaGModel
from models.delta_G.cnn_deltaG_prediction import CNNDeltaGModel
from models.delta_G.gat_network import GATCrossAttentionPretrain, GATCrossAttentionPretrainPI
import time
import torch
import numpy as np
import random
import os

def setup_seed(seed):
    torch.manual_seed(seed)
setup_seed(20)


identity = f"KGML_KG"
model = GATDeltaGModel(
    train_path=f"data/deltaG_metal_train.csv",
    val_path="data/deltaG_metal_val.csv",
    test_path="data/deltaG_metal_test.csv",
    ckpt_path=f"checkpoints/delta_G_{identity}",
    log_path=f"record/delta_G_{identity}",
    in_channels=30,
    gat_hidden_channels=256,
    gat_layers=8,
    out_channels=1,
    output_hidden_dim=512,
    bs=64,
    lr=1e-4,
    key_dim=256,
    value_dim=256,
    metal_embed_dim=256,
    query_dim=256,
    network=GATCrossAttentionPretrainPI, # For Model-Base please choose GATCrossAttentionPretrain
    uncertainty=False,
    PI=True, # For Model-Base please choose False
    pretrain_path='pretrain_model/model_epoch_100.pt',
    ext=False, # For Model-KGE please choose True
)

model.fit(num_epochs=250, p=150)

os.makedirs("results", exist_ok=True)

r = model.predict_train_test(
    result_train_path=f"results/delta_G_train_{identity}.csv",
    result_test_path=f"results/delta_G_test_{identity}.csv",
    result_val_path=f"results/delta_G_val_{identity}.csv"
)

print("Train R2:", r[0][0])
print("Val R2:", r[0][1])
print("Test R2:", r[0][2])
print("Train mse:", r[1][0])
print("Val mse:", r[1][1])
print("Test mse:", r[1][2])
print("Train Uncertainty:", r[2][0])
print("Val Uncertainty:", r[2][1])
print("Test Uncertainty:", r[2][2])
