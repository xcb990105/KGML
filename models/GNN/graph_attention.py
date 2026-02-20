import torch.nn as nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F

class PAWLayer(nn.Module):
    """
    Attention mechanism for node and virtual node embedding.
    """
    def __init__(self, in_channels, out_channels, dropout=0, edge_dim=None):
        super(PAWLayer, self).__init__()
        if dropout > 0:
            self.attentive_layer = GATConv(in_channels, out_channels, dropout=dropout, edge_dim=edge_dim)
        else:
            self.attentive_layer = GATConv(in_channels, out_channels, edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr=None):
        # Attention-based node feature transformation
        return F.relu(self.attentive_layer(x, edge_index, edge_attr=edge_attr))

class ResidualGATLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=3, dropout=0.2):
        super(ResidualGATLayer, self).__init__()
        self.gat_conv_1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat_conv_2 = GATConv(hidden_channels * heads, out_channels, heads=1, dropout=dropout)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        residual = self.residual(x)
        out = self.gat_conv_1(x, edge_index)
        out = self.gat_conv_2(out, edge_index)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)
        out += residual
        return out