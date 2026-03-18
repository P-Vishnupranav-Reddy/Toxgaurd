#!/usr/bin/env python3
"""
==============================================================================
ToxGuard Phase 2 — EGNN Pipeline
Step 03: E(n) Equivariant Graph Neural Network — Model Architecture
==============================================================================

Reference Paper:
    "E(n) Equivariant Graph Neural Networks"
    Victor Garcia Satorras, Emiel Hoogeboom, Max Welling (ICML 2021)
    arXiv: 2102.09844

Architecture Overview:
    The EGNN operates on molecular graphs with 3D coordinates. It updates
    both node features (h) and coordinates (x) at each layer while maintaining
    E(n) equivariance — the output transforms correctly under rotations,
    translations, and reflections of the input coordinates.

    Key equations from the paper (Eqs. 3-6):
        m_ij  = φ_e(h_i, h_j, ||x_i - x_j||², a_ij)     (edge message)
        x_i'  = x_i + C Σ_j (x_i - x_j) · φ_x(m_ij)     (coordinate update)
        m_i   = Σ_j m_ij                                   (aggregate messages)
        h_i'  = φ_h(h_i, m_i)                              (node update)

    where a_ij are optional edge attributes (bond features in our case).

Design Decisions for Toxicity Prediction:
    1. We include edge attributes (bond type, conjugation, ring membership,
       stereo) as they carry crucial chemical information.
    2. We use coordinate updates (not just invariant features) as the paper
       shows this improves performance.
    3. Graph-level readout uses attention-weighted pooling for better
       expressiveness than simple mean/sum pooling.
    4. The classification head uses dropout and batch normalisation for
       regularisation on our relatively small datasets.
    5. We add a "virtual node" mechanism to improve global information flow —
       important for toxicity which can depend on distant substructures.

This file is importable as a module by the training script (Step 04).

Author: ToxGuard Team
==============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import softmax as pyg_softmax
from typing import Optional, Tuple


# ============================================================================
# EGNN Layer  (Equations 3-6 from the paper)
# ============================================================================
class EGNNLayer(nn.Module):
    """
    Single E(n) Equivariant Graph Neural Network layer.
    
    Implements the message-passing scheme from Satorras et al. (2021):
    
        m_ij  = φ_e(h_i ∥ h_j ∥ ||x_i - x_j||² ∥ a_ij)
        x_i'  = x_i + (1/|N(i)|) Σ_{j∈N(i)} (x_i - x_j) · φ_x(m_ij)
        m_i   = Σ_{j∈N(i)} m_ij
        h_i'  = h_i + φ_h(h_i ∥ m_i)                      [residual connection]
    
    Modifications for molecular toxicity prediction:
        - Edge attributes (bond features) are included in the edge model
        - Coordinate updates are normalised to prevent coordinate explosion
        - Residual connections on node features for stable deep networks
        - Layer normalisation for training stability
    
    Args:
        hidden_dim:    Dimension of node feature vectors h
        edge_feat_dim: Dimension of edge attribute vectors a_ij
        act_fn:        Activation function (default: SiLU/Swish)
        update_coords: Whether to update coordinates (set False for last layer)
        dropout:       Dropout rate in MLPs
        norm_coords:   Whether to normalise coordinate displacements
        tanh_coords:   Apply tanh to coordinate updates (prevents explosion)
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        edge_feat_dim: int = 0,
        act_fn: nn.Module = nn.SiLU(),
        update_coords: bool = True,
        dropout: float = 0.0,
        norm_coords: bool = True,
        tanh_coords: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.update_coords = update_coords
        self.norm_coords = norm_coords
        self.tanh_coords = tanh_coords
        
        # Edge model: φ_e
        # Input: h_i (hidden_dim) + h_j (hidden_dim) + d²_ij (1) + a_ij (edge_feat_dim)
        edge_input_dim = 2 * hidden_dim + 1 + edge_feat_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        
        # Node model: φ_h
        # Input: h_i (hidden_dim) + m_i (hidden_dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Coordinate model: φ_x (outputs scalar weight per edge)
        if update_coords:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                act_fn,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1, bias=False),
            )
            # Initialize coord_mlp output near zero for stability
            nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.001)
        
        # Layer normalisation for node features
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        h: Tensor,           # (N, hidden_dim) node features
        x: Tensor,           # (N, 3) coordinates
        edge_index: Tensor,  # (2, E) edge indices
        edge_attr: Optional[Tensor] = None,  # (E, edge_feat_dim)
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for one EGNN layer.
        
        Returns:
            h_out: Updated node features (N, hidden_dim)
            x_out: Updated coordinates (N, 3)
        """
        row, col = edge_index  # row = source, col = target
        
        # Compute squared distances ||x_i - x_j||²
        coord_diff = x[row] - x[col]  # (E, 3)
        dist_sq = (coord_diff ** 2).sum(dim=-1, keepdim=True)  # (E, 1)
        
        # --- Edge messages: m_ij = φ_e(h_i, h_j, d²_ij, a_ij) ---
        edge_input = torch.cat([h[row], h[col], dist_sq], dim=-1)
        if edge_attr is not None:
            edge_input = torch.cat([edge_input, edge_attr], dim=-1)
        
        m_ij = self.edge_mlp(edge_input)  # (E, hidden_dim)
        
        # --- Coordinate update: x_i' = x_i + Σ_j (x_i - x_j) · φ_x(m_ij) ---
        if self.update_coords:
            coord_weights = self.coord_mlp(m_ij)  # (E, 1)
            
            if self.tanh_coords:
                coord_weights = torch.tanh(coord_weights)
            
            if self.norm_coords:
                # Normalise displacement vectors to unit length
                dist = torch.sqrt(dist_sq + 1e-8)  # (E, 1)
                coord_diff_norm = coord_diff / dist  # (E, 3)
                weighted_diff = coord_diff_norm * coord_weights  # (E, 3)
            else:
                weighted_diff = coord_diff * coord_weights  # (E, 3)

            # Keep scatter-add dtype consistent under AMP.
            weighted_diff = weighted_diff.to(dtype=x.dtype)
            
            # Aggregate coordinate updates
            x_update = torch.zeros_like(x)
            x_update.index_add_(0, row, weighted_diff)
            
            # Normalise by node degree
            degree = torch.zeros(x.size(0), 1, device=x.device)
            degree.index_add_(0, row, torch.ones(row.size(0), 1, device=x.device))
            degree = degree.clamp(min=1)
            
            x_out = x + x_update / degree
        else:
            x_out = x
        
        # --- Message aggregation: m_i = Σ_j m_ij ---
        # Keep scatter-add dtype consistent under AMP.
        m_ij = m_ij.to(dtype=h.dtype)
        m_i = torch.zeros_like(h)
        m_i.index_add_(0, row, m_ij)
        
        # --- Node update: h_i' = h_i + φ_h(h_i, m_i) --- (residual)
        node_input = torch.cat([h, m_i], dim=-1)
        h_out = h + self.node_mlp(node_input)
        h_out = self.layer_norm(h_out)
        
        return h_out, x_out


# ============================================================================
# Attention-Weighted Graph Pooling
# ============================================================================
class AttentionPooling(nn.Module):
    """
    Attention-weighted global pooling for graph-level readout.
    
    Instead of simple mean/sum pooling, learn attention weights per node
    to focus on atoms most relevant for toxicity prediction.
    
    This is important because toxicity often depends on specific functional
    groups (toxicophores) rather than the entire molecule uniformly.
    
    Args:
        hidden_dim: Dimension of node features
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, h: Tensor, batch: Tensor) -> Tensor:
        """
        Args:
            h:     (N_total, hidden_dim) node features from all graphs in batch
            batch: (N_total,) graph membership for each node
        
        Returns:
            graph_emb: (B, hidden_dim) graph-level embeddings
        """
        # Compute attention scores
        attn_scores = self.attention(h)  # (N_total, 1)
        
        # Softmax within each graph
        attn_weights = pyg_softmax(attn_scores, batch)  # (N_total, 1)
        
        # Weighted sum
        weighted_h = h * attn_weights  # (N_total, hidden_dim)
        graph_emb = global_add_pool(weighted_h, batch)  # (B, hidden_dim)
        
        return graph_emb


# ============================================================================
# Full EGNN Model for Toxicity Prediction
# ============================================================================
class ToxEGNN(nn.Module):
    """
    E(n) Equivariant Graph Neural Network for molecular toxicity prediction.
    
    Architecture:
        1. Input projection: atom features → hidden dimension
        2. Edge projection: bond features → edge hidden dimension
        3. N stacked EGNN layers (equivariant message passing)
        4. Attention-weighted graph pooling → molecule embedding
        5. Classification head: MLP → toxicity probability
    
    The model is E(n)-equivariant in its intermediate representations and
    E(n)-invariant at the output (as required for scalar prediction).
    
    Args:
        node_feat_dim:  Input atom feature dimension (from Step 01)
        edge_feat_dim:  Input bond feature dimension (from Step 01)
        hidden_dim:     Hidden dimension for EGNN layers
        num_layers:     Number of EGNN message-passing layers
        dropout:        Dropout rate
        update_coords:  Update coordinates in EGNN layers
        pool_method:    'attention', 'mean', or 'sum'
        num_classes:    Number of output classes (1 for binary)
    """
    
    def __init__(
        self,
        node_feat_dim: int = 58,
        edge_feat_dim: int = 12,
        hidden_dim: int = 256,
        num_layers: int = 6,
        dropout: float = 0.1,
        update_coords: bool = True,
        pool_method: str = "attention",
        num_classes: int = 1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pool_method = pool_method
        
        # --- Input projections ---
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.SiLU(),
        ) if edge_feat_dim > 0 else None
        
        # Edge hidden dim for EGNN layers
        edge_hidden = hidden_dim if edge_feat_dim > 0 else 0
        
        # --- Stacked EGNN layers ---
        self.egnn_layers = nn.ModuleList()
        for i in range(num_layers):
            # Don't update coords in last layer (we only need invariant output)
            update_this_layer = update_coords and (i < num_layers - 1)
            self.egnn_layers.append(
                EGNNLayer(
                    hidden_dim=hidden_dim,
                    edge_feat_dim=edge_hidden,
                    act_fn=nn.SiLU(),
                    update_coords=update_this_layer,
                    dropout=dropout,
                    norm_coords=True,
                    tanh_coords=True,
                )
            )
        
        # --- Graph-level readout ---
        if pool_method == "attention":
            self.pooling = AttentionPooling(hidden_dim)
        elif pool_method == "mean":
            self.pooling = None  # Use global_mean_pool
        elif pool_method == "sum":
            self.pooling = None  # Use global_add_pool
        else:
            raise ValueError(f"Unknown pool_method: {pool_method}")
        
        # --- Classification head ---
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        
        # Initialise weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier uniform initialisation for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, data: Data) -> Tensor:
        """
        Forward pass.
        
        Args:
            data: PyG Data or Batch object with:
                - x:          (N, node_feat_dim) atom features
                - pos:        (N, 3) coordinates
                - edge_index: (2, E) bond graph
                - edge_attr:  (E, edge_feat_dim) bond features
                - batch:      (N,) graph membership indices
        
        Returns:
            logits: (B, 1) raw logits for binary classification
        """
        h = self.node_encoder(data.x)       # (N, hidden_dim)
        x = data.pos                         # (N, 3)
        edge_index = data.edge_index         # (2, E)
        
        # Encode edge attributes
        edge_attr = None
        if self.edge_encoder is not None and data.edge_attr is not None:
            edge_attr = self.edge_encoder(data.edge_attr)  # (E, hidden_dim)
        
        # --- EGNN message passing ---
        for layer in self.egnn_layers:
            h, x = layer(h, x, edge_index, edge_attr)
        
        # --- Graph-level readout ---
        batch = data.batch if hasattr(data, "batch") and data.batch is not None \
            else torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        
        if self.pool_method == "attention":
            graph_emb = self.pooling(h, batch)
        elif self.pool_method == "mean":
            graph_emb = global_mean_pool(h, batch)
        else:
            graph_emb = global_add_pool(h, batch)
        
        # --- Classification ---
        logits = self.classifier(graph_emb)  # (B, 1)
        
        return logits
    
    def get_embeddings(self, data: Data) -> Tensor:
        """
        Extract molecule-level embeddings (before classification head).
        Useful for analysis, t-SNE visualisation, or fusion with Phase 1.
        
        Returns:
            embeddings: (B, hidden_dim)
        """
        h = self.node_encoder(data.x)
        x = data.pos
        edge_index = data.edge_index
        
        edge_attr = None
        if self.edge_encoder is not None and data.edge_attr is not None:
            edge_attr = self.edge_encoder(data.edge_attr)
        
        for layer in self.egnn_layers:
            h, x = layer(h, x, edge_index, edge_attr)
        
        batch = data.batch if hasattr(data, "batch") and data.batch is not None \
            else torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        
        if self.pool_method == "attention":
            graph_emb = self.pooling(h, batch)
        elif self.pool_method == "mean":
            graph_emb = global_mean_pool(h, batch)
        else:
            graph_emb = global_add_pool(h, batch)
        
        return graph_emb
    
    def count_parameters(self) -> dict:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ============================================================================
# Model factory
# ============================================================================
def create_model(
    node_feat_dim: int,
    edge_feat_dim: int,
    config: Optional[dict] = None,
) -> ToxEGNN:
    """
    Create a ToxEGNN model with the given configuration.
    
    Args:
        node_feat_dim: Dimension of input atom features
        edge_feat_dim: Dimension of input bond features
        config: Optional dict with model hyperparameters
    
    Returns:
        ToxEGNN model
    """
    defaults = {
        "hidden_dim": 256,
        "num_layers": 6,
        "dropout": 0.1,
        "update_coords": True,
        "pool_method": "attention",
        "num_classes": 1,
    }
    
    if config is not None:
        defaults.update(config)
    
    model = ToxEGNN(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        **defaults,
    )
    
    return model


# ============================================================================
# Test the model (run this file directly to verify)
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ToxEGNN Architecture Verification")
    print("=" * 60)
    
    # Create a small synthetic batch for testing
    torch.manual_seed(42)
    
    node_feat_dim = 58  # From Step 01 featurization
    edge_feat_dim = 12  # From Step 01 featurization
    
    # Two small molecules (5 and 3 atoms)
    data1 = Data(
        x=torch.randn(5, node_feat_dim),
        pos=torch.randn(5, 3),
        edge_index=torch.tensor([[0,1,1,2,2,3,3,4], [1,0,2,1,3,2,4,3]], dtype=torch.long),
        edge_attr=torch.randn(8, edge_feat_dim),
        y=torch.tensor([1.0]),
    )
    data2 = Data(
        x=torch.randn(3, node_feat_dim),
        pos=torch.randn(3, 3),
        edge_index=torch.tensor([[0,1,1,2], [1,0,2,1]], dtype=torch.long),
        edge_attr=torch.randn(4, edge_feat_dim),
        y=torch.tensor([0.0]),
    )
    
    batch = Batch.from_data_list([data1, data2])
    
    # Create model
    model = create_model(node_feat_dim, edge_feat_dim)
    
    print(f"\nModel architecture:")
    print(model)
    
    params = model.count_parameters()
    print(f"\nParameter count:")
    print(f"  Total:     {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    
    # Forward pass
    logits = model(batch)
    print(f"\nForward pass:")
    print(f"  Input:  batch of {batch.num_graphs} graphs")
    print(f"  Output: logits shape = {logits.shape}")
    print(f"  Logits: {logits.detach().squeeze().tolist()}")
    
    probs = torch.sigmoid(logits)
    print(f"  Probs:  {probs.detach().squeeze().tolist()}")
    
    # Test equivariance: rotate input and check output is invariant
    print("\nE(n) Invariance test:")
    # Random rotation matrix
    theta = torch.tensor(0.5)
    R = torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0],
        [torch.sin(theta),  torch.cos(theta), 0],
        [0, 0, 1],
    ], dtype=torch.float32)
    
    # Apply rotation to coordinates
    batch_rotated = batch.clone()
    batch_rotated.pos = batch.pos @ R.T
    
    logits_original = model(batch).detach()
    logits_rotated = model(batch_rotated).detach()
    
    diff = (logits_original - logits_rotated).abs().max().item()
    print(f"  Max absolute difference after rotation: {diff:.6f}")
    print(f"  Invariance {'PASSED' if diff < 0.01 else 'FAILED'} (threshold: 0.01)")
    
    # Test translation invariance
    batch_translated = batch.clone()
    batch_translated.pos = batch.pos + 100.0
    
    logits_translated = model(batch_translated).detach()
    diff_trans = (logits_original - logits_translated).abs().max().item()
    print(f"  Max absolute difference after translation: {diff_trans:.6f}")
    print(f"  Translation invariance {'PASSED' if diff_trans < 0.01 else 'FAILED'}")
    
    # Embeddings
    emb = model.get_embeddings(batch)
    print(f"\nEmbeddings shape: {emb.shape}")
    
    print("\n" + "=" * 60)
    print("Architecture verification complete.")
    print("=" * 60)
