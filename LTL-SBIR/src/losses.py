
import torch
import torch.nn as nn
import torch.nn.functional as F

def pairwise_distances(embeddings: torch.Tensor) -> torch.Tensor:
    """Pairwise L2 distances between rows of embeddings."""
    dot = embeddings @ embeddings.t()
    sq = torch.diag(dot)
    dist = sq.unsqueeze(1) - 2.0 * dot + sq.unsqueeze(0)
    dist = torch.clamp(dist, min=0.0)
    # numerical stability for sqrt
    mask = (dist == 0.0).float()
    dist = torch.sqrt(dist + mask * 1e-16) * (1.0 - mask)
    return dist

class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, p=2, dim=1)
        D = pairwise_distances(embeddings)
        N = labels.size(0)
        labels = labels.view(N, 1)
        mask_pos = torch.eq(labels, labels.t()).to(embeddings.device)
        mask_neg = ~mask_pos
        mask_pos = mask_pos ^ torch.eye(N, dtype=torch.bool, device=embeddings.device)

        pos_dist = D * mask_pos.float()
        hardest_pos = pos_dist.max(dim=1)[0]

        neg_dist = D + 1e5 * (~mask_neg).float()
        hardest_neg = neg_dist.min(dim=1)[0]

        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        return loss.mean()
