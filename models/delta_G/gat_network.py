import torch.nn as nn
from models.GNN.graph_attention import PAWLayer
from torch_geometric.nn import global_mean_pool
from models.GNN.cross_attention import CrossAttention
import torch.nn.functional as F
import torch



class GATCrossAttention(nn.Module):
    def __init__(self, node_in_channels, hidden_channels, out_channels, num_node_layers=4,
                 metal_feature_dim=8, metal_embed_dim=128, query_dim=128, key_dim=128, value_dim=128, output_hidden_dim=512):
        super(GATCrossAttention, self).__init__()
        self.node_in_channels = node_in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        # Node embedding layers
        self.node_layers = nn.ModuleList([
            PAWLayer(-1, hidden_channels, dropout=0, edge_dim=11)
            for _ in range(num_node_layers)
        ])

        self.metal_fc = nn.Sequential(
            nn.Linear(metal_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, metal_embed_dim),
        )

        # Cross Attention layer
        self.cross_attention = CrossAttention(query_dim, key_dim, value_dim, hidden_channels=hidden_channels)

        self.regressor = nn.Sequential(
            nn.Linear(value_dim + metal_embed_dim, output_hidden_dim),
            nn.ReLU(),
            nn.Linear(output_hidden_dim, output_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_hidden_dim, output_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

        # Fully connected layer for output
        self.fc_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, data, metal_features):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        # Node embedding stage
        for layer in self.node_layers:
            x = layer(x, edge_index, edge_attr=edge_attr)
            # x = F.elu(x)

        metal_embed = self.metal_fc(metal_features)

        key = x  # K: (num_nodes, hidden_channels)
        value = x  # V: (num_nodes, hidden_channels)

        # Apply cross-attention
        attn_output = self.cross_attention(metal_embed, key, value, batch)  # (batch_size, value_dim)

        combined = torch.cat([attn_output, metal_embed], dim=1)

        energy = self.regressor(combined).squeeze(-1)

        return energy

class GATCrossAttentionPretrain(nn.Module):
    def __init__(self, node_in_channels, hidden_channels, out_channels, num_node_layers=4,
                 metal_feature_dim=8, metal_embed_dim=128, query_dim=128, key_dim=128, value_dim=128,
                 output_hidden_dim=512, p_hidden_channels=256, p_num_node_layers=15):
        super(GATCrossAttentionPretrain, self).__init__()
        self.node_in_channels = node_in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.node_layers = nn.ModuleList([
            PAWLayer(-1, hidden_channels, dropout=0.2, edge_dim=11)
            for _ in range(num_node_layers)
        ])

        self.gat_layers = nn.ModuleList([
            PAWLayer(-1, hidden_channels, edge_dim=11)
            for _ in range(num_node_layers)
        ])

        self.metal_fc = nn.Sequential(
            nn.Linear(metal_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, metal_embed_dim),
        )

        # Cross Attention layer
        self.cross_attention = CrossAttention(query_dim, key_dim, value_dim, hidden_channels=hidden_channels)

        self.regressor = nn.Sequential(
            nn.Linear(value_dim + metal_embed_dim, output_hidden_dim),
            nn.ReLU(),
            nn.Linear(output_hidden_dim, output_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_hidden_dim, output_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        self.regressor[2].register_forward_hook(self._hook_fn)
        # Fully connected layer for output
        self.fc_out = nn.Linear(hidden_channels, out_channels)

    def load_pretrained_weights(self, pretrain_path, device):
        print("loading pretrained weights...")
        pretrain_state_dict = torch.load(pretrain_path, map_location=device)

        model_state_dict = self.state_dict()
        for name, param in pretrain_state_dict.items():
            if name.startswith('node_layers'):
                model_state_dict[name] = param

        self.load_state_dict(model_state_dict, strict=False)
        self.to(device)

    def _hook_fn(self, module, input, output):
        self.intermediate_output = output

    def forward(self, data, metal_features, export_hidden_feature=False):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        # Node embedding stage

        f = x
        for layer in self.node_layers:
            f = layer(f, edge_index, edge_attr=edge_attr)

        x = torch.cat([x, f], dim=1)

        for layer in self.gat_layers:
            x = layer(x, edge_index, edge_attr=edge_attr)

        metal_embed = self.metal_fc(metal_features)

        key = x  # K: (num_nodes, hidden_channels)
        value = x  # V: (num_nodes, hidden_channels)

        # Apply cross-attention
        attn_output = self.cross_attention(metal_embed, key, value, batch)  # (batch_size, value_dim)

        combined = torch.cat([attn_output, metal_embed], dim=1)

        energy = self.regressor(combined).squeeze(-1)

        if export_hidden_feature:
            return energy, self.intermediate_output
        else:
            return energy

class GATCrossAttentionPI(nn.Module):
    def __init__(self, node_in_channels, hidden_channels, out_channels, num_node_layers=4,
                 metal_feature_dim=8, metal_embed_dim=128, query_dim=128, key_dim=128, value_dim=128, output_hidden_dim=512):
        super(GATCrossAttentionPI, self).__init__()
        self.node_in_channels = node_in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        # Node embedding layers
        self.node_layers = nn.ModuleList([
            PAWLayer(-1, hidden_channels)
            for _ in range(num_node_layers)
        ])

        self.global_pool = global_mean_pool

        self.metal_fc = nn.Sequential(
            nn.Linear(metal_feature_dim, metal_embed_dim),
            nn.ReLU(),
            nn.Linear(metal_embed_dim, metal_embed_dim),
            nn.ReLU()
        )

        # Cross Attention layer
        self.cross_attention = CrossAttention(query_dim, key_dim, value_dim, hidden_channels=hidden_channels)

        self.regressor = nn.Sequential(
            nn.Linear(10 + value_dim + metal_embed_dim, output_hidden_dim),
            nn.ReLU(),
            nn.Linear(output_hidden_dim, output_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_hidden_dim, output_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

        self.pi_fc = nn.Sequential(
            nn.Linear(hidden_channels, output_hidden_dim),
            nn.ReLU(),
            nn.Linear(output_hidden_dim, output_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_hidden_dim, output_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 10)
        )

        # Fully connected layer for output
        self.fc_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, data, metal_features, pi=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Node embedding stage
        for layer in self.node_layers:
            x = layer(x, edge_index)
            x = F.elu(x)

        pi_feature = self.global_pool(x, batch)

        if pi:
            pi_out = self.pi_fc(pi_feature)
            return pi_out
        else:
            metal_embed = self.metal_fc(metal_features)

            key = x  # K: (num_nodes, hidden_channels)
            value = x  # V: (num_nodes, hidden_channels)

            pi_out = self.pi_fc(pi_feature)
            # Apply cross-attention
            attn_output = self.cross_attention(metal_embed, key, value, batch)  # (batch_size, value_dim)

            combined = torch.cat([pi_out, attn_output, metal_embed], dim=1)

            energy = self.regressor(combined).squeeze(-1)

            return energy, pi_out

class GATCrossAttentionPretrainPI(nn.Module):
    def __init__(self, node_in_channels, hidden_channels, out_channels, num_node_layers=4,
                 metal_feature_dim=8, metal_embed_dim=128, query_dim=128, key_dim=128, value_dim=128, output_hidden_dim=512):
        super(GATCrossAttentionPretrainPI, self).__init__()
        self.node_in_channels = node_in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        # Node embedding layers
        self.node_layers = nn.ModuleList([
            PAWLayer(-1, hidden_channels, dropout=0.2, edge_dim=None)
            for _ in range(num_node_layers)
        ])

        self.gat_layers = nn.ModuleList([
            PAWLayer(-1, hidden_channels, edge_dim=None)
            for _ in range(num_node_layers)
        ])

        self.global_pool = global_mean_pool

        self.metal_fc = nn.Sequential(
            nn.Linear(metal_feature_dim, metal_embed_dim),
            nn.ReLU(),
            nn.Linear(metal_embed_dim, metal_embed_dim),
            nn.ReLU()
        )

        # Cross Attention layer
        self.cross_attention = CrossAttention(query_dim, key_dim, value_dim, hidden_channels=hidden_channels)

        self.regressor = nn.Sequential(
            nn.Linear(10 + value_dim + metal_embed_dim, output_hidden_dim),
            nn.ReLU(),
            nn.Linear(output_hidden_dim, output_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_hidden_dim, output_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        self.regressor[2].register_forward_hook(self._hook_fn)

        self.pi_fc = nn.Sequential(
            nn.Linear(hidden_channels, output_hidden_dim),
            nn.ReLU(),
            nn.Linear(output_hidden_dim, output_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_hidden_dim, output_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 10)
        )

        # Fully connected layer for output
        self.fc_out = nn.Linear(hidden_channels, out_channels)

    def _hook_fn(self, module, input, output):
        self.intermediate_output = output

    def load_pretrained_weights(self, pretrain_path, device):
        print("loading pretrained weights...")
        pretrain_state_dict = torch.load(pretrain_path, map_location=device)

        model_state_dict = self.state_dict()
        for name, param in pretrain_state_dict.items():
            if name.startswith('node_layers'):
                model_state_dict[name] = param

        self.load_state_dict(model_state_dict, strict=False)
        self.to(device)

    def forward(self, data, metal_features, pi=False, get_attention=False, export_hidden_feature=False):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        # Node embedding stage
        f = x
        for layer in self.node_layers:
            f = layer(f, edge_index, edge_attr=None)

        x = torch.cat([x, f], dim=1)

        for layer in self.gat_layers:
            x = layer(x, edge_index, edge_attr=None)

        pi_feature = self.global_pool(x, batch)

        if pi:
            pi_out = self.pi_fc(pi_feature)
            return pi_out
        else:
            metal_embed = self.metal_fc(metal_features)

            key = x  # K: (num_nodes, hidden_channels)
            value = x  # V: (num_nodes, hidden_channels)

            pi_out = self.pi_fc(pi_feature)
            # Apply cross-attention
            if get_attention:
                attn_output, attention_weights = self.cross_attention(metal_embed, key, value, batch, get_attention=True)
            else:
                attn_output = self.cross_attention(metal_embed, key, value, batch)  # (batch_size, value_dim)

            combined = torch.cat([pi_out, attn_output, metal_embed], dim=1)

            energy = self.regressor(combined).squeeze(-1)
            if get_attention:
                return energy, pi_out, attention_weights
            if export_hidden_feature:
                return energy, pi_out, self.intermediate_output
            else:
                return energy, pi_out

