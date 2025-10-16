import torch
import torch.nn as nn
import torch_geometric as thg
import torch_geometric.nn as gnn
from torch.distributions.categorical import Categorical


def layer_init(layer, std: float = 2.0 ** 0.5, bias_const: float = 0.0):
    """
    Initialize a linear layer with orthogonal weights and constant bias.

    Args:
        layer: Linear layer to initialize
        std: Standard deviation for weight initialization
        bias_const: Constant value for bias initialization

    Returns:
        Initialized layer
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class HGT(nn.Module):
    """
    Heterogeneous Graph Transformer (HGT) encoder for multi-type node and edge graphs.

    Processes heterogeneous graph data with different node types (units, servers)
    and relation types (in, contains, moveto, etc.).
    """

    def __init__(
            self,
            metadata,
            hidden_channels: int,
            out_channels: int,
            num_heads: int,
            num_layers: int
    ):
        super().__init__()

        # Initial linear projection for each node type
        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = gnn.Linear(-1, hidden_channels)

        # Stack of HGTConv layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = gnn.HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            self.convs.append(conv)

        # Final output projection per node type
        self.out_lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            self.out_lin_dict[node_type] = gnn.Linear(hidden_channels, hidden_channels)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        """
        Forward pass through the HGT encoder.

        Args:
            x_dict: Dictionary mapping node types to feature tensors
            edge_index_dict: Dictionary mapping edge types to edge index tensors

        Returns:
            Transformed x_dict after message passing
        """
        # Initial projection and ReLU activation
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        # Message passing through HGT layers
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        # Final linear transformation
        x_dict = {
            node_type: self.out_lin_dict[node_type](x)
            for node_type, x in x_dict.items()
        }

        return x_dict


class ActorNetwork(nn.Module):
    """
    Actor network that outputs actions (unit_id, server_id) using a two-stage selection process.

    Uses GNN embeddings and masked softmax to ensure only valid actions are selected.
    """

    def __init__(self, device: torch.device):
        super().__init__()
        self.hidden_channels = 128
        self.num_heads = 2
        self.num_layers = 4
        self.num_units = 300
        self.num_servers = 30
        self.device = device

        self.metadata = (
            ['units', 'src_servers', 'dst_servers'],
            [
                ('units', 'in', 'src_servers'),
                ('src_servers', 'contains', 'units'),
                ('units', 'moveto', 'dst_servers'),
                ('dst_servers', 'contains', 'units'),
                ('dst_servers', 'correspond', 'src_servers'),
                ('src_servers', 'rcorrespond', 'dst_servers'),
            ]
        )

        self.graphenc = HGT(
            metadata=self.metadata,
            hidden_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            num_heads=self.num_heads,
            num_layers=self.num_layers
        )

        # Two-stage action head
        self.fc_units = nn.Sequential(
            layer_init(nn.Linear(self.hidden_channels, self.hidden_channels)),
            nn.ReLU(),
            layer_init(nn.Linear(self.hidden_channels, 1))
        )
        self.fc_servers = nn.Sequential(
            layer_init(nn.Linear(self.hidden_channels * 2, self.hidden_channels * 2)),
            nn.ReLU(),
            layer_init(nn.Linear(self.hidden_channels * 2, 1))
        )
        self.tanh = nn.Tanh()

    def forward(
            self,
            obs: thg.data.Batch,
            edge_candidates: list,
            action: tuple = (None, None)
    ) -> tuple:
        """
        Compute action log-probabilities and entropy for PPO training.

        Args:
            obs: Batched heterogeneous graph observation
            edge_candidates: List of valid server candidates for each unit
            action: Optional tuple (act1, act2) for on-policy evaluation

        Returns:
            (actions, log_probs, entropy)
        """
        x_dict, edge_index_dict, batch_dict = obs.x_dict, obs.edge_index_dict, obs.batch_dict
        x_dict = self.graphenc(x_dict, edge_index_dict)
        return self._act(x_dict, batch_dict, edge_candidates, action)

    def eval_actlist(
            self,
            obs_list: list,
            edge_candidates: list,
            k1: int = 3,
            k2: int = 2
    ) -> list:
        """
        Evaluate top-k valid actions during inference.

        Args:
            obs_list: List of graph observations
            edge_candidates: List of valid actions per unit
            k1: Number of top units to consider
            k2: Number of top servers per unit

        Returns:
            List of top-(k1*k2) actions as (unit_id, server_id)
        """
        batch = thg.data.Batch.from_data_list(obs_list).to(self.device)
        x_dict, edge_index_dict, batch_dict = batch.x_dict, batch.edge_index_dict, batch.batch_dict
        x_dict = self.graphenc(x_dict, edge_index_dict)

        bs = len(edge_candidates)
        mask1, _ = self._get_mask1(bs, self.num_units, edge_candidates)
        units_x, units_batch = x_dict['units'], batch_dict['units']
        servers_x, servers_batch = x_dict['src_servers'], batch_dict['src_servers']

        # Score all units
        unit_scores = self.tanh(self.fc_units(units_x)).squeeze(-1)
        unit_scores, _ = thg.utils.to_dense_batch(unit_scores, units_batch, max_num_nodes=self.num_units)
        unit_probs = nn.functional.softmax(unit_scores + mask1, dim=1)
        top_unit_indices = unit_probs.argsort(dim=-1, descending=True)[:k1]

        # Expand features for candidate scoring
        units_dense, _ = thg.utils.to_dense_batch(units_x, units_batch, max_num_nodes=self.num_units)
        servers_dense, _ = thg.utils.to_dense_batch(servers_x, servers_batch, max_num_nodes=self.num_servers)

        action_list = []
        for unit_idx in top_unit_indices:
            if mask1[0, unit_idx] == -float('inf'):
                continue

            mask2, _ = self._get_mask2(bs, self.num_servers, edge_candidates, [unit_idx.item()])
            unit_feat = units_dense[0, unit_idx]  # Assume batch size 1
            expanded_unit = unit_feat.unsqueeze(0).unsqueeze(1).expand(1, self.num_servers, -1)
            combined = torch.cat([servers_dense[0], expanded_unit], dim=-1)

            server_scores = self.tanh(self.fc_servers(combined)).squeeze(-1)
            server_probs = nn.functional.softmax(server_scores + mask2[0], dim=0)
            top_server_indices = server_probs.argsort(descending=True)[:k2]

            for server_idx in top_server_indices:
                if mask2[0, server_idx] > -float('inf'):
                    action_list.append((unit_idx, server_idx))

        return action_list

    def _act(
            self,
            x_dict: dict,
            batch_dict: dict,
            edge_candidates: list,
            action: tuple
    ) -> tuple:
        """Internal method to compute action distribution."""
        bs = len(edge_candidates)
        mask1, _ = self._get_mask1(bs, self.num_units, edge_candidates)
        act1, logp1, ent1 = self._select(x_dict, batch_dict, mask1, action[0])

        mask2, _ = self._get_mask2(bs, self.num_servers, edge_candidates, act1)
        act2, logp2, ent2 = self._select(x_dict, batch_dict, mask2, action[1], act1)

        return torch.stack((act1, act2)), torch.stack((logp1, logp2)), ent1 + ent2

    def _select(
            self,
            x_dict: dict,
            batch_dict: dict,
            mask: torch.Tensor,
            action: torch.Tensor,
            prev_action: torch.Tensor = None
    ) -> tuple:
        """Select action from logits with masking."""
        units_x, units_batch = x_dict['units'], batch_dict['units']
        servers_x, servers_batch = x_dict['src_servers'], batch_dict['src_servers']

        if prev_action is None:
            # Select unit
            x = self.fc_units(units_x).squeeze(-1)
            x = self.tanh(x) * 10
            x, _ = thg.utils.to_dense_batch(x, units_batch, max_num_nodes=self.num_units)
        else:
            # Select server given unit
            units_dense, _ = thg.utils.to_dense_batch(units_x, units_batch, max_num_nodes=self.num_units)
            servers_dense, _ = thg.utils.to_dense_batch(servers_x, servers_batch, max_num_nodes=self.num_servers)
            unit_feat = units_dense[torch.arange(len(prev_action)), prev_action]
            expanded = unit_feat.unsqueeze(1).expand(-1, servers_dense.shape[1], -1)
            x = torch.cat([servers_dense, expanded], dim=-1)
            x = self.fc_servers(x).squeeze(-1)
            x = self.tanh(x) * 10

        probs = nn.functional.softmax(x + mask, dim=1)
        dist = Categorical(probs=probs)
        sampled_action = dist.sample() if action is None else action
        return sampled_action, dist.log_prob(sampled_action), dist.entropy()

    def _get_mask1(self, batch_size: int, num_nodes: int, edge_candidates: list) -> tuple:
        """Generate mask for first action (unit selection)."""
        mask = torch.full((batch_size, num_nodes), -float('inf'), device=self.device)
        for b in range(batch_size):
            for node_id, candidates in enumerate(edge_candidates[b]):
                if candidates:
                    mask[b, node_id] = 0.0
        return mask, None

    def _get_mask2(self, batch_size: int, num_nodes: int, edge_candidates: list, act1: torch.Tensor) -> tuple:
        """Generate mask for second action (server selection)."""
        mask = torch.full((batch_size, num_nodes), -float('inf'), device=self.device)
        for b in range(batch_size):
            unit_id = act1[b].item()
            candidates = edge_candidates[b][unit_id]
            for idx in candidates:
                mask[b, idx] = 0.0
        return mask, None


class CriticNetwork(nn.Module):
    """
    Critic network that estimates state value using global pooling over GNN embeddings.
    """

    def __init__(self, device: torch.device):
        super().__init__()
        self.hidden_channels = 128
        self.num_heads = 2
        self.num_layers = 4
        self.device = device

        self.metadata = (
            ['units', 'src_servers', 'dst_servers'],
            [
                ('units', 'in', 'src_servers'),
                ('src_servers', 'contains', 'units'),
                ('units', 'moveto', 'dst_servers'),
                ('dst_servers', 'contains', 'units'),
                ('dst_servers', 'correspond', 'src_servers'),
                ('src_servers', 'rcorrespond', 'dst_servers'),
            ]
        )

        self.graphenc = HGT(
            metadata=self.metadata,
            hidden_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            num_heads=self.num_heads,
            num_layers=self.num_layers
        )

        self.mlp = nn.Sequential(
            layer_init(nn.Linear(self.hidden_channels * 3, self.hidden_channels)),
            nn.ReLU(),
            layer_init(nn.Linear(self.hidden_channels, 1), std=1.0)
        )

    def forward(self, obs: thg.data.Batch) -> torch.Tensor:
        """Estimate the value of the current state."""
        x_dict, edge_index_dict, batch_dict = obs.x_dict, obs.edge_index_dict, obs.batch_dict
        x_dict = self.graphenc(x_dict, edge_index_dict)

        units_pooled = gnn.global_mean_pool(x_dict['units'], batch_dict['units'])
        src_pooled = gnn.global_mean_pool(x_dict['src_servers'], batch_dict['src_servers'])
        dst_pooled = gnn.global_mean_pool(x_dict['dst_servers'], batch_dict['dst_servers'])

        concatenated = torch.cat((units_pooled, src_pooled, dst_pooled), dim=1)
        return self.mlp(concatenated)


class Agent(nn.Module):
    """
    Full RL agent combining actor and critic networks.
    """

    def __init__(self, device: torch.device):
        super().__init__()
        self.actor = ActorNetwork(device)
        self.critic = CriticNetwork(device)
        self.device = device

    def get_value(self, obs_list: list) -> torch.Tensor:
        """Get value estimate for a list of observations."""
        batch = thg.data.Batch.from_data_list(obs_list).to(self.device)
        return self.critic(batch)

    def get_action(self, obs_list: list, edge_candidates: list, action: tuple = (None, None)):
        """
        Get action, log-prob, and entropy for training or inference.

        Args:
            obs_list: List of graph observations
            edge_candidates: Valid action masks per unit
            action: Optional pre-defined action for on-policy learning

        Returns:
            (action, log_prob, entropy)
        """
        batch = thg.data.Batch.from_data_list(obs_list).to(self.device)
        return self.actor(batch, edge_candidates, action)
