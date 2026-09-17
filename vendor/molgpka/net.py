"""Plain-PyTorch reimplementation of MolGpKa's `utils/net.py::GCNNet` (Xundrug/MolGpKa, MIT).

The original depends on `torch_geometric.nn.GCNConv`/`GlobalAttention` and
`torch_scatter.scatter_add`. `torch_scatter` is a compiled extension pinned to an
exact torch build with no reliable prebuilt wheel for Apple Silicon -- installing it
here would mean a from-source compile for a single 60-line layer. This file
reproduces that layer's exact math in plain `torch.Tensor` operations instead, so the
weights (`weight_acid.pth`, `weight_base.pth`, unmodified from upstream) load into an
architecturally identical module.

Only `GCNNet` is reproduced -- upstream's `GATNet` is unused by `predict_pka.py`, and
`MPNNNet` is dead code in the original repo (references undefined `Sequential`,
`NNConv`, `GRU`, `Set2Set` -- it does not run there either).

**This file is an adaptation, not a pin.** Unlike `vendor/cyp_challenge_tutorial/`,
which is copied verbatim because downstream code must match the leaderboard's own
scoring bit-for-bit, nothing here compares against MolGpKa's original output --
only against chemical plausibility (see `tests/test_pka.py`). See `vendor/README.md`
for why this vendor entry is held to a different standard.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

N_FEATURES = 29
HIDDEN = 1024


def _glorot(tensor: torch.Tensor) -> None:
    """Upstream's `utils/inits.py::glorot` -- a uniform init scaled by fan-in/out,
    not `torch.nn.init.xavier_uniform_`'s gain-adjusted variant. The state_dict was
    trained under this initialisation's resulting parameter scale, so matching the
    *shape* of the loaded weights (done by `load_state_dict`) is what matters here,
    not the init itself -- this function exists only so a freshly constructed
    `GCNConvPT` has sane values before a state_dict is loaded onto it."""
    stdv = (6.0 / (tensor.size(-2) + tensor.size(-1))) ** 0.5
    tensor.data.uniform_(-stdv, stdv)


class GCNConvPT(nn.Module):
    """Kipf & Welling's GCN propagation rule, matching `torch_geometric`'s
    `GCNConv` (`gcn_conv.py` upstream) parameter-for-parameter: `weight` and `bias`
    are the only learnable tensors, so a state_dict trained against the PyG layer
    loads here unchanged.

    `torch_scatter.scatter_add` is replaced by `torch.zeros(...).index_add_`, which
    computes the identical sum-of-neighbours aggregation without the compiled
    extension. `add_remaining_self_loops` (from `torch_geometric.utils`) is
    reproduced directly: append a self-loop edge for every node not already carrying
    one, so the same symmetric-normalised adjacency (`D^-1/2 (A+I) D^-1/2`) is built.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        _glorot(self.weight)

    @staticmethod
    def _normalise(edge_index: torch.Tensor, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
        device = edge_index.device
        # Self-loops: every node gets one, once -- upstream's `add_remaining_self_loops`
        # with `fill_value=1` (GCNConv's default, `improved=False`).
        self_loops = torch.arange(num_nodes, device=device)
        row = torch.cat([edge_index[0], self_loops])
        col = torch.cat([edge_index[1], self_loops])
        weight = torch.ones(row.size(0), device=device)

        deg = torch.zeros(num_nodes, device=device).index_add_(0, row, weight)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        norm = deg_inv_sqrt[row] * weight * deg_inv_sqrt[col]
        return torch.stack([row, col]), norm

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = x @ self.weight
        (row, col), norm = self._normalise(edge_index, x.size(0))
        # `message` + `aggr='add'`: each edge contributes `norm * x[col]` (the
        # neighbour's transformed features) into `row` (the receiving node).
        messages = norm.view(-1, 1) * x[col]
        out = torch.zeros_like(x).index_add_(0, row, messages)
        return out + self.bias


class GlobalAttentionPT(nn.Module):
    """Matches `torch_geometric.nn.GlobalAttention(gate_nn=Linear(hidden, 1))`: a
    learned scalar gate per node, softmax-normalised *within* each graph in the
    batch, then a weighted sum of node features. `GCNNet` calls this with a single
    linear gate and no separate feature transform (`nn=None` in the original), so
    only `gate_nn`'s weight/bias are learned parameters here.
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.gate_nn = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        gate = self.gate_nn(x)  # (n_nodes, 1)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        out = torch.zeros(num_graphs, x.size(1), device=x.device)
        for g in range(num_graphs):
            mask = batch == g
            weights = F.softmax(gate[mask], dim=0)
            out[g] = (weights * x[mask]).sum(dim=0)
        return out


class GCNNet(nn.Module):
    """Reproduces upstream's `GCNNet`: 5 GCN layers with BatchNorm + ReLU, global
    attention pooling to one vector per molecule, then a 3-layer MLP to a scalar.
    Parameter names match upstream exactly (`conv1..5`, `bn1..5`, `att`, `fc2..4`)
    so `load_state_dict` on the original `.pth` files works without remapping.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = GCNConvPT(N_FEATURES, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.conv2 = GCNConvPT(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.conv3 = GCNConvPT(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = GCNConvPT(256, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.conv5 = GCNConvPT(512, 1024)
        self.bn5 = nn.BatchNorm1d(1024)

        self.att = GlobalAttentionPT(HIDDEN)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = self.att(x, batch)

        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
