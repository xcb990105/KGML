import torch.nn as nn
import torch
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, hidden_channels):
        super(CrossAttention, self).__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim

        # Linear layers for Q, K, V
        self.query_proj = nn.Linear(query_dim, key_dim)  # Project Q to match K's dimension
        self.key_proj = nn.Linear(hidden_channels, key_dim)       # Project K
        self.value_proj = nn.Linear(hidden_channels, value_dim) # Project V

        # Output projection
        self.out_proj = nn.Linear(value_dim, value_dim)

    def forward(self, query, key, value, batch, get_attention=False):

        # Project Q, K, V
        query = self.query_proj(query)  # (batch_size, key_dim)
        key = self.key_proj(key)        # (num_nodes, key_dim)
        value = self.value_proj(value)  # (num_nodes, value_dim)

        # Initialize output
        out = torch.zeros_like(query)  # (batch_size, value_dim)
        attention_weights = []
        # Iterate over each graph in the batch
        for graph_idx in range(query.size(0)):
            # Get the nodes belonging to the current graph
            node_mask = (batch == graph_idx)  # (num_nodes,)
            node_indices = torch.where(node_mask)[0]  # node_id
            graph_key = key[node_mask]       # (num_nodes_in_graph, key_dim)
            graph_value = value[node_mask]    # (num_nodes_in_graph, value_dim)

            # Compute attention scores for the current graph
            scores = torch.matmul(query[graph_idx].unsqueeze(0), graph_key.transpose(0, 1)) / (self.key_dim ** 0.5)  # (1, num_nodes_in_graph)
            attn_weights = F.softmax(scores, dim=-1)  # (1, num_nodes_in_graph)

            # Apply attention weights to values
            graph_out = torch.matmul(attn_weights, graph_value)  # (1, value_dim)
            out[graph_idx] = graph_out.squeeze(0)  # (value_dim,)
            if get_attention:
                attention_weights.append({
                    "graph_idx": graph_idx, 
                    "node_ids": node_indices.cpu().numpy(), 
                    "attention_weights": attn_weights.squeeze(0).detach().cpu().numpy()
                })

        # Project output
        out = self.out_proj(out)  # (batch_size, value_dim)
        if get_attention:
            return out, attention_weights
        return out
