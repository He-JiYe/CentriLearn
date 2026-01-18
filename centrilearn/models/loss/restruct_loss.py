import torch
import torch.nn.functional as F


# 重建损失
def reconstruction_loss(z, edge_index, device="cuda"):
    adj_pred = torch.sigmoid(z @ z.t())
    adj_true = torch.sparse_coo_tensor(
        edge_index, torch.ones(edge_index.shape[1], device=device), size=adj_pred.shape
    ).to_dense()
    return F.binary_cross_entropy(adj_pred, adj_true)
