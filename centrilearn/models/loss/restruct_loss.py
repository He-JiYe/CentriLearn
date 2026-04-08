"""
Graph Reconstruction Loss Functions
"""

import torch
import torch.nn.functional as F


def reconstruction_loss(z, edge_index, batch_ptr, device="cuda"):
    """Reconstruction loss for batched graphs.

    Args:
        z: Node embeddings [num_nodes, embedding_dim]
        edge_index: Edge indices [2, num_edges]
        batch_ptr: Batch boundary pointers [num_graphs + 1]
        device: Device to use

    Returns:
        Reconstruction loss value
    """
    num_graphs = len(batch_ptr) - 1
    total_loss = 0.0
    num_nodes_total = 0

    for i in range(num_graphs):
        start, end = batch_ptr[i].item(), batch_ptr[i + 1].item()
        num_nodes = end - start

        if num_nodes == 0:
            continue

        z_graph = z[start:end]
        adj_pred = torch.sigmoid(z_graph @ z_graph.t())

        edges_mask = (
            (edge_index[0] >= start)
            & (edge_index[0] < end)
            & (edge_index[1] >= start)
            & (edge_index[1] < end)
        )
        local_edges = edge_index[:, edges_mask] - start
        adj_true = torch.zeros(num_nodes, num_nodes, device=device)
        adj_true[local_edges[0], local_edges[1]] = 1.0

        total_loss += F.binary_cross_entropy(adj_pred, adj_true) * num_nodes
        num_nodes_total += num_nodes

    return (
        total_loss / num_nodes_total
        if num_nodes_total > 0
        else torch.tensor(0.0, device=device)
    )
