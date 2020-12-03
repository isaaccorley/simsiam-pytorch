import torch
import torch.nn.functional as F


def negative_cosine_similarity(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    z = z.detach()
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return - (p * z).sum(dim=1).mean()

def simsiam_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor
) -> torch.Tensor:
    loss1 = negative_cosine_similarity(p1, z1)
    loss2 = negative_cosine_similarity(p2, z2)
    return loss1/2 + loss2/2
